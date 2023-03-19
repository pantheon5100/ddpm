from denoising_diffusion_pytorch import UnetLCAtten, GaussianDiffusion, Trainer
import torch

def main():
    model = UnetLCAtten(
        dim = 64,
        dim_mults = (1, 2, 4, 8)
    )
    # model = torch.compile(model)

    diffusion = GaussianDiffusion(
        model,
        image_size = 32,
        timesteps = 1000,           # number of steps
        sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type = 'l1',            # L1 or L2
        beta_schedule = 'sigmoid',
    )

    trainer = Trainer(
        diffusion,
        './data',
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 700000,         # total training steps
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = False,                       # turn on mixed precision
        calculate_fid = True,              # whether to calculate fid during training
        results_folder = './LCAtten_results'
    )

    trainer.train()


if __name__ == "__main__":
    main()

