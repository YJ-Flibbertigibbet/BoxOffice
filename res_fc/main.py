from pandas.core.arrays.sparse.array import splib
import train

#====================================
def main(split=True):
    train.resTrain(split=split)


#====================================

if __name__== "__main__":
    main(split=False)