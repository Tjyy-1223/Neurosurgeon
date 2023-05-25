import torch
from predictor_utils import *
from utils import inference_utils






if __name__ == '__main__':
    predictor_dict = {}
    x = torch.rand((1,3,224,224))
    # x = torch.rand((1, 512, 28, 28))
    # model = inference_utils.get_dnn_model("vgg_net")
    model = inference_utils.get_dnn_model("mobile_net")
    # print(model)
    # model = nn.Conv2d(512,512,kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    model.eval()

    # print(predict_layer_latency(model,x,edge_device=False,show=True))
    for layer in model:
        if skip_layer(layer):
            x = layer(x)
            continue
        # lat = predict_layer_latency(layer,x,edge_device=True,show=True)
        lat = predict_model_latency(x,layer,device="cloud",predictor_dict=predictor_dict)
        x = layer(x)
        print(f"{layer} , latency: {lat:.2f} ms ,  x.shape : {x.shape}")
        print("============================================")






