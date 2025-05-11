import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from NMF import spa, tensor_unfold_m3
from INR import Siren, get_mgrid, SLFFitting_Irregular, fibersampling_exclude
from Metrics import Scores

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
vis_bin = 16
MapFiles = 'Radiomapseer_R2_IRT4.mat'

X = np.array(loadmat(MapFiles)['X_GT'])
Xmode3 = tensor_unfold_m3(X)
R = 2
space1, space2, freq = X.shape
l2_coefficient = 1e-16

samplingsize = 1500
location_files = 'building_location_index.mat'
buildinglocations = np.array(loadmat(location_files)['building_location_index'], dtype=int)  # get building locations

Yob, sampleindex = fibersampling_exclude(X, samplingsize, buildinglocations)

Yob_tensor = torch.from_numpy(Yob).float().to(device)
sel_idx = spa(Yob, R)
Gslf = Yob[:, sel_idx]
col_max = np.diag(np.max(Gslf, axis=0))
Gslf = np.dot(Gslf, np.diag(1 / np.diag(col_max)))
P = np.transpose(np.dot(np.linalg.pinv(Gslf), Yob), (1, 0))

models = [Siren(in_features=2, out_features=1, hidden_features=256,
                hidden_layers=4, outermost_linear=True).to(device) for _ in range(R)]

optimizers = [torch.optim.Adam(lr=1e-4, params=model.parameters()) for model in models]

coor_inputs = []
slf_labels = []

for r in range(R):
    Gslf_norm = (Gslf[:, r] - 0.5) / 0.5
    slfvec = SLFFitting_Irregular(Gslf_norm, max(space1, space2), sampleindex)
    dataloader = DataLoader(slfvec, batch_size=1, pin_memory=True, num_workers=0)
    model_input, ground_truth = next(iter(dataloader))
    coor_inputs.append(model_input.to(device))
    slf_labels.append(ground_truth.to(device))

total_steps = 150

grid_hyper_resolution = get_mgrid(256, 2)
grid_hyper_resolution = grid_hyper_resolution.unsqueeze(0).to(device)

RMScore = Scores()
"two-stage initialization"
for step in range(total_steps):
    G_reconstruct = np.zeros([space1 * space2, R])
    for r, (model, coor, label, optimizer) in enumerate(zip(models, coor_inputs, slf_labels, optimizers)):
        output, coords = model(coor)
        loss = ((output - label) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{step + 1}/{total_steps}], INR {r + 1}, Loss: {loss.item():.4f}')

        with torch.no_grad():
            slf_hyper_resolution, hyper_coords = model.forward(grid_hyper_resolution)
            # de-normalized
            slf_hyper_resolution_dn = slf_hyper_resolution * 0.5 + 0.5
            G_reconstruct_col = slf_hyper_resolution_dn.squeeze()
            G_reconstruct[:, r] = G_reconstruct_col.cpu().numpy()
    X_reconstruct_mode3 = np.dot(G_reconstruct, np.transpose(P, (1, 0)))
    X_reconstruct = np.reshape(X_reconstruct_mode3, [space1, space2, freq])
    X_reconstruct = np.transpose(X_reconstruct, (1, 0, 2))

    NMSE = RMScore.compute_nmse(X, X_reconstruct)
    SSIM = RMScore.compute_ssim(X, X_reconstruct)

    print(f'Epoch [{step + 1}/{total_steps}], NMSE: {NMSE:.4f}, SSIM: {SSIM: .4f}')

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    im0 = axes[0].imshow(X[:, :, vis_bin], cmap='jet')
    axes[0].set_title('Ground-truth', fontsize=16)
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(X_reconstruct[:, :, vis_bin], cmap='jet')
    axes[1].set_title(f'GFRME\nNMSE={NMSE:.4f}, SSIM={SSIM:.4f}', fontsize=16)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
    plt.close()

"closed-loop update"
Yob_tensor = Yob_tensor.unsqueeze(0)

# for r in range(R):
#     optimizers[r] = torch.optim.Adam(params=models[r].parameters(), lr=1e-6)

closed_loop_steps = 5
inner_steps = 40

P_closed_loop_update = torch.from_numpy(P).float()
P_closed_loop_update = P_closed_loop_update.permute(1, 0).to(device)

for step in range(closed_loop_steps):
    G_reconstruct = np.zeros([space2 * space1, R])
    for step_inner in range(inner_steps):
        G_step = []
        for r, (model, coor) in enumerate(zip(models, coor_inputs)):
            output_r, coords = model(coor)
            output_r = output_r * 0.5 + 0.5
            G_step.append(output_r.squeeze(0))

        G_step_to_matrix = torch.cat(G_step, dim=1)
        predicted_output = G_step_to_matrix @ P_closed_loop_update
        predicted_output = predicted_output.unsqueeze(0)
        loss = ((predicted_output - Yob_tensor) ** 2).mean()
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
    with torch.no_grad():
        reg_term = l2_coefficient * torch.eye(R, device=device)
        G_T_G_pinv = torch.linalg.pinv(G_step_to_matrix.permute(1, 0) @ G_step_to_matrix + reg_term)
        P_closed_loop_update = G_T_G_pinv @ (G_step_to_matrix.permute(1, 0) @ Yob_tensor.squeeze())
    print(f'Epoch [{step + 1}/{closed_loop_steps}], Closed-Loop Loss: {loss.item():.4f}')

    with torch.no_grad():
        for r in range(R):
            slf_hyper_resolution, hyper_coords = models[r].forward(grid_hyper_resolution)
            slf_hyper_resolution_dn = slf_hyper_resolution * 0.5 + 0.5
            G_reconstruct_col = slf_hyper_resolution_dn.squeeze()
            G_reconstruct[:, r] = G_reconstruct_col.cpu().numpy()

    X_reconstruct_mode3 = np.dot(G_reconstruct, P_closed_loop_update.cpu().numpy())

    X_reconstruct = np.reshape(X_reconstruct_mode3, [space1, space2, freq])
    X_reconstruct = np.transpose(X_reconstruct, (1, 0, 2))

    NMSE = RMScore.compute_nmse(X, X_reconstruct)
    SSIM = RMScore.compute_ssim(X, X_reconstruct)

    print(f'Epoch [{step + 1}/{closed_loop_steps}], NMSE: {NMSE:.4f}, SSIM: {SSIM: .4f}')

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    im0 = axes[0].imshow(X[:, :, vis_bin], cmap='jet')
    axes[0].set_title('Ground-truth', fontsize=16)
    axes[0].axis('off')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(X_reconstruct[:, :, vis_bin], cmap='jet')
    axes[1].set_title(f'GFRME\nNMSE={NMSE:.4f}, SSIM={SSIM:.4f}', fontsize=16)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(1)
    plt.close()


mdicXhat = {"Xgfrme": X_reconstruct, "NMSEgfrme": NMSE, 'SSIMgfrme': SSIM}
savemat("RadioIRT4_INR_1500.mat", mdicXhat)
