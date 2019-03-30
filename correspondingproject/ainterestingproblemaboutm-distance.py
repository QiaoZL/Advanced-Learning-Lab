# Matt Aliabadi
# matt@avvo.com
#

# Each square on a grid of indeterminate size is allocated in a spiral pattern starting
# at a location marked 1 and then counting up while spiraling outward. For example, the
# first few squares are allocated like this:
# 
# 37  36  35  34  33  32  31
# 38  17  16  15  14  13  30
# 39  18   5   4   3  12  29
# 40  19   6   1   2  11  28
# 41  20   7   8   9  10  27
# 42  21  22  23  24  25  26
# 43  44  45  46  47  48  49
#
# Implement a method for finding the Manhattan Distance (# of moves up, down, left, or right)
# between any two given points on the grid.
#
#


# 1, 12 -> 3
# 9, 49 -> 4

#round[i] = round[i-1] + 4
#diff(round[i][first],round[i][last]) = diff(round[i-1][first],round[i-1][last])+8
#i>=1
#for any number n, there may have: n = f(x,y,i)
#find the function f should be the best solution.
#may also have method can calculate distance for (x,y)

#direction of x->y, if we can find it we can find the best algorithm, key is the 8, the step difference from round and round and 1, the step difference from same round

#there are only two types of step, +-8 or +-1, unless one of (x,y) is 1

#center is the key problem, are there method can deal with the center?

#we can divide the matrix into 4 different parts, corss parts or not can have diffrent calculation method, may simply add the distance based on properties of m-distance

#basic difference from round 1 to round 2 is problem

def stepwisemod(x,y,w=8):
    
    diff = x-y
    i=1
    stepsize = 0
    
    while(abs(diff-stepsize)<8):
        stepsize += 8*i
        i+=1
        


def compare(x,y):
    
    #should corret when x and y in same part
    
    if x==y:
        return 0
    
    diff = x-y
    print(diff)
    stepssame = 0
    stepsdiff = 0

    stepsdiff_small = diff//8
    stepsdiff_large = diff//8 + 1
    
    t1 = abs(stepsdiff_small*8 - diff)
    t2 = abs(stepsdiff_large*8 - diff)
    
    if(t1>=t2):
        stepsdiff = stepsdiff_large
    else:
        stepsdiff = stepsdiff_small
    
    print(stepsdiff)
    
    stepssame = diff - stepsdiff*8
    
    print(stepssame)
    
    
    return(abs(stepssame + stepsdiff))

def trans(x):
    
    stepsize = []
    minstep = 0
    
    for i in range(2,10):
        stepsize.append(compare(i,x))
        
    minstep = stepsize.index(min(stepsize))
        
    
        

print(compare(9,49))
 
    