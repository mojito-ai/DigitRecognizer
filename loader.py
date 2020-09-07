import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import network2
net= network2.Network([784,30,10])
cost=network2.CrossEntropyCost()
net.SGD(training_data[:10000],30,10,1.0,lmbda=20,evaluation_data=validation_data[:100],
        monitor_evaluation_accuracy=True)
