import neural_predict
import neural_train
import neural_validate
import datetime

def array_set():
    predictions =[]
    for x in range(2000-1900, 2017-1900):
        for y in range(1, 12):
            temp=(x*12+y)
            predictions.append(temp)
            print(predictions)

def main():
    numberofsteps=5000
    for x in range(0,30):
        neural_train.train(numberofsteps)
        neural_validate.validate()
        neural_predict.predict()
        #array_set()



if __name__ == '__main__':
    main()


