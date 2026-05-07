class AgroTinyNet_v2(nn.Module):
    """
    The Upgraded Student Model: Featuring a U-Net style skip connection 
    to preserve high-resolution spatial details for tiny objects.
    """
    def __init__(self, num_classes=3): 
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )
        
        # Encoder
        self.layer1 = DepthwiseSeparableConv(16, 32, stride=2)  # High-res edges
        self.layer2 = DepthwiseSeparableConv(32, 64, stride=2)  
        self.layer3 = DepthwiseSeparableConv(64, 128, stride=1) # Low-res semantics
        
        # Decoder (Fusion layer for the skip connection)
        # We concatenate 128 channels (deep) + 32 channels (skip) = 160
        self.fusion = DepthwiseSeparableConv(160, 64, stride=1)
        
        # Final Segmentation Head
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.size()[2:] 
        
        x = self.stem(x)
        skip = self.layer1(x) 
        
        x = self.layer2(skip)
        features = self.layer3(x) # Deep semantic features
        
        # Upsample the deep features to match the physical size of our saved skip features
        features_up = F.interpolate(features, size=skip.size()[2:], mode='bilinear', align_corners=False)
        
        # Concatenate them together
        fused = torch.cat([features_up, skip], dim=1) 
        fused = self.fusion(fused)
        
        out = self.classifier(fused)
        
        # Final resize back to 224x224
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        return out, features