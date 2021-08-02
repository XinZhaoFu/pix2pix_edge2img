import argparse


class Parse_Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='get args')

    def parseArgs(self):
        """
        获得参数

        :return:
        """

        self.parser.add_argument('--learning_rate',
                                 dest='learning_rate',
                                 help='learning_rate',
                                 default=0,
                                 type=float)
        self.parser.add_argument('--epochs',
                                 dest='epochs',
                                 help='epochs',
                                 default=1,
                                 type=int)
        self.parser.add_argument('--batch_size',
                                 dest='batch_size',
                                 help='batch_size',
                                 default=4,
                                 type=int)
        self.parser.add_argument('--load_train_file_number',
                                 dest='load_train_file_number',
                                 help='load_train_file_number',
                                 default=0,
                                 type=int)
        self.parser.add_argument('--load_val_file_number',
                                 dest='load_val_file_number',
                                 help='load_val_file_number',
                                 default=0,
                                 type=int)
        self.parser.add_argument('--load_weights',
                                 dest='load_weights',
                                 help='load_weights type is boolean',
                                 default=False, type=bool)
        self.parser.add_argument('--data_augmentation',
                                 dest='data_augmentation',
                                 help='data_augmentation type is float, range is 0 ~ 1',
                                 default=0,
                                 type=float)
        args = self.parser.parse_args()
        return args
