x=1
y=2.0
z="Here is a string"


p=1.0f0
q=Float32(1.0)
typeof(p),typeof(q)

r=1.0

t=(x,y,z)

struct House
    Address::String
    Size::Float32
    Construction::String
end

MyHouse=House("Australia",100.0,"Brick")

a=[1.0, 2.0 ,3.0,4.0, 88888.0]
aa=[1.0  2.0  3.0 4.0 88888.0]

b=[2.0 3.0 4.0;8.0 9.0 4.0]

x=0.0:0.1:1.0


a=[1,2,3]
b=[2.0 3.0 4.0;8.0 9.0 4.0]
b*a



a=[1.0, 2.0 ,3.0]
f(x)=x^3
f(a)
f.(a)

A=[1 2 3;4 5 6;7 8 9]
C=[2, 3, 4]
x=A\C #equivalent to A^{-1}*C
A*x-C

function my_first_function(x,y)
    mean=0.5*(x+y)
end

my_first_function(1.0,2.0)

f(x)=x^3
dummy_func=(x,p)->f(x)



x=2
function my_print(x)
    println(x)
end

my_print(x)


x=2
function my_print2(b)
    println(x)
end


x=7.0
for i=1:5
    x-=1.0
end


xoutsideloop=3.0

for i=1:3
    xoutsideloop+=1.0
    xinsideloop=xoutsideloop+1.0
end