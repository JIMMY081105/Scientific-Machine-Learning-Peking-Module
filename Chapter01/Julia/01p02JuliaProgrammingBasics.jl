x=1
y=2.0
z="Here is a string"

typeof(x)

p=1.0f0
q=Float32(1.0)
typeof(p),typeof(q)

t=(x,y,z)

typeof(t)

t[3]

struct House
    Address::String
    Size::Float32
    Construction::String
end

MyHouse=House("Australia",100.0,"Brick")

MyHouse.Address

MyHouse.Size=200

# Define a mutable struct called Counter
mutable struct Age
    Years::Int
end
MyAge=Age(70)

MyAge.Years=60

MyAge.Years

Vector{Float64}(undef, 4)

Matrix{Float64}(undef, 2, 4)

a=[1.0, 2.0 ,3.0]

b=[2.0 3.0 4.0;8.0 9.0 4.0]

b[1,:]

b[:,2]

transpose(b)

b'

x=0.0:0.1:1.0

Array(x)

xrange=-1:1:1
yrange=1:1:3
loss_values=[x+y^2 for x in xrange, y in yrange]

a=1.0+2.0 # addition
b=1.0-2.0 #subtraction
c=2.0*3.0 #multiplication
d=3.0^3 #power

(a,b,c,d)


1.0/2.0

1.0\2.0 

a=[1,2,3]
b=[2.0 3.0 4.0;8.0 9.0 4.0]
b*a

b*a

a.*a

b.*b

a.*b[1,:]

A=[1 2 3;4 5 6;7 8 9]
C=[2, 3, 4]
x=A\C #equivalent to A^{-1}*C
A*x-C

f(x)=x^3

typeof(f(1)),typeof(f(2.0))

f.(a)

function my_first_function(x,y)
    mean=0.5*(x+y)
end

my_first_function(1.0,2.0)


dummy_func=(x,p)->f(x)
dummy_func(3,10) 

x=2
function my_print(x)
    println(x)
end

my_print(3)

x=2
function my_print(b)
    println(x)
end

my_print(10)

x=7.0
for i=1:5
    x-=1.0
end
x

xoutsideloop=3.0

for i=1:3
    xoutsideloop+=1.0
    xinsideloop=xoutsideloop+1.0
end

xoutsideloop

xinsideloop
