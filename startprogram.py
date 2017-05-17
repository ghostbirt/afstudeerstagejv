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
    numberofsteps=50
    for x in range(0,1):
        neural_train.train(numberofsteps)
        neural_validate.validate(numberofsteps/10)
        neural_predict.predict()
        #array_set()



if __name__ == '__main__':
    main()


