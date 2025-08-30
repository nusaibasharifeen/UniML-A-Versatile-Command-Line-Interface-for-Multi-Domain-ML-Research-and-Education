from torch import device


def get_model(name):
    if name == "dcgan":
        from .dcgan import DCGAN
        return DCGAN()
    elif name == "cgan":
        from .cgan import ConditionalGAN
        return ConditionalGAN(noise_dim=100,num_classes=10,img_channels=1,img_size=28)
    elif name == "cvae":
        from .cvae import ConditionalVAE
        return ConditionalVAE(num_classes=10,img_channels=1,img_size=28,latent_dim=20,beta=1.0)
    elif name == "gpt2":
        from .gpt2 import GPT2FineTuner
        return GPT2FineTuner(
            model_name='gpt2',
            output_dir='./DATA/output',
            per_device_train_batch_size=2,
            num_train_epochs=30.0,
            save_steps=500,
            max_length=100
        )
    elif name == "wavegan":
        from .wavegan import WaveGANGenerator, WaveGANDiscriminator
        LATENT_DIM = 100
        AUDIO_LENGTH = 16384
        return WaveGANGenerator(latent_dim=LATENT_DIM,audio_length=AUDIO_LENGTH).to(device), WaveGANDiscriminator(audio_length=AUDIO_LENGTH).to(device)
    else:
        raise ValueError(f"Unknown model: {name}")
