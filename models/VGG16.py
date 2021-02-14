from torch.nn import Module, Conv2d, MaxPool2d, Linear, ReLU, Sequential


class ConvBlock(Module):
    def __init__(self, input_channels, output_channels, no_of_conv):
        super().__init__()
        layer_list = []
        for i in range(no_of_conv):
            layer_list.append(Conv2d(in_channels=input_channels if i==0 else output_channels, 
                                    out_channels=output_channels, 
                                    kernel_size=(3, 3), 
                                    stride = 1, 
                                    padding = 1))
            layer_list.append(ReLU())
        layer_list.append(MaxPool2d(kernel_size=(2, 2), stride=2, padding=0))
        self.block = Sequential(*layer_list)

    def forward(self, X):
        return self.block(X)


class VGG16(Module):
    def __init__(self, no_of_classes):
        super().__init__()
        self.block1 = ConvBlock(3, 64, 2)
        self.block2 = ConvBlock(64, 128, 2)
        self.block3 = ConvBlock(128, 256, 3)
        self.block4 = ConvBlock(256, 512, 3)
        self.block5 = ConvBlock(512, 512, 3)

        self.Linear1 = Linear(7*7*512, 4096)
        self.Linear2 = Linear(4096, 4096)
        self.Linear3 = Linear(4096, no_of_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.shape[0], -1)
        x = self.Linear1(x)
        x = self.Linear2(x)
        x = self.Linear3(x)
        return x