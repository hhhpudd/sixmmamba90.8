from models_mamba import VisionMamba

print(VisionMamba(img_size=14, patch_size=1, embed_dim=384, depth=24, rms_norm=True))