import sys

#==========================================
# Differentiation
#==========================================
# J. Douglas Faiers / Richard L. Burden
# "Numerische Methoden" p. 168
# http://www.trentfguidry.net/post/2010/09/04/Numerical-differentiation-formulas.aspx
def d5_0(f,h):
    tmp = -2.0833333*f[0]+4.0*f[1]-3.0*f[2]+1.3333333*f[3]-0.25*f[4]
    return tmp/h

def d5_1(f,h):
    tmp = -0.25*f[0]-0.8333333*f[1]+1.5*f[2]-0.5*f[3]+0.08333333*f[4]
    return tmp/h
    
def d5_2(f,h):
    tmp = 0.0833333*f[0]-0.666666*f[1]+0.666666*f[3]-0.0833333*f[4]
    return tmp/h

def d5_3(f,h):
    tmp = -0.0833333*f[0]+0.5*f[1]-1.5*f[2]+0.8333333*f[3]+0.25*f[4]
    return tmp/h

def d5_4(f,h):
    tmp = 0.25*f[0]-1.333333*f[1]+3.0*f[2]-4.0*f[3]+2.08333333*f[4]
    return tmp/h
#==========================================

def derivative(x,y):
    """
    Use as standalone, give x and y data
    y will be differentiated and returned
    x: list, numpy_array 1D
    y: list, numpy_array 1D
    returns: list
    """
    t = list(x)
    f = list(y)
    d = []
    h = t[1]-t[0]
    lenf = len(f)
    d.append(d5_0(f[0:5],h))
    d.append(d5_1(f[0:5],h))
    for i in range(lenf-5):
        d.append(d5_2(f[i:i+5],h))
    d.append(d5_3(f[i:i+5],h))
    d.append(d5_4(f[i:i+5],h))

    return d


def main():
    #==========================================
    # Reading data
    #==========================================
    datafile = sys.argv[1]
    file=open(datafile,"r")
    t = []
    f = []
    for line in file:
        if len(line) < 2:
            break
        line_element = line.split()
        t.append(float(line_element[0]))
        f.append(float(line_element[1]))
    
    #d = []
    #h = t[1]-t[0]
    #lenf = len(f)
    #d.append(d5_0(f[0:5],h))
    #d.append(d5_1(f[0:5],h))
    #for i in range(lenf-5):
    #    d.append(d5_2(f[i:i+5],h))
    #d.append(d5_3(f[i:i+5],h))
    #d.append(d5_4(f[i:i+5],h))
    d = derivative(t,f)
    for i in range(len(d)):
        print ("%10.5f %10.5f" % (t[i],d[i]))

if __name__ == "__main__":
    main()
