from tensorboardX import SummaryWriter

class Visualization:
    def __init__(self):
        self.writer = ''#= SummaryWriter(logdir=model_type, comment=model_type)

    def create_summary(self, model_type='U_Net'):
        """新建writer 设置路径"""
        # self.writer = SummaryWriter(model_type, comment=model_type)
        self.writer = SummaryWriter(comment='-' +model_type)

    def add_scalar(self, epoch, value, params='loss'):
        """添加训练记录"""
        self.writer.add_scalar(params, value, global_step=epoch)

    def add_iamge(self, tag, img_tensor):
        """添加tensor影像"""
        self.writer.add_iamge(tag, img_tensor)

    def add_graph(self, model):
        """添加模型图"""
        self.writer.add_graph(model)

    def close_summary(self):
        """关闭writer"""
        self.writer.close()