import neural_predict
import neural_train
import neural_validate


def main():
    numberofsteps = 500
    for x in range(0, 20):
        neural_train.train(numberofsteps)
        neural_validate.validate()
        neural_predict.predict()


if __name__ == '__main__':
    main()
