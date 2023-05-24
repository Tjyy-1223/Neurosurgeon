from utils.inference_utils import *


if __name__ == '__main__':
    # arg = "alex_net"
    arg = "mobile_net"

    model = get_dnn_model(arg)
    input_data = torch.rand(size=(1,3,224,224))
    show_features(model,input_data,device="cpu")