#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace std;

class Shape
{
protected:
    int x;
    int y;

public:
    Shape()
    {
        srand(time(0));
        x = rand() % 100;
        y = rand() % 100;
    }
    Shape(int x, int y)
    {
        this->x = x;
        this->y = y;
    }
    void input()
    {
        cout << "Enter x and y: ";
        cin >> x >> y;
    }
    void output()
    {
        cout << "x = " << x << " y = " << y << endl;
    }

    virtual double getPer() const = 0;

    virtual ~Shape(){};
};

class Line : public Shape
{
private:
    double len;
    int x2;
    int y2;

public:
    Line() : Shape()
    {
        x2 = rand() % 100;
        y2 = rand() % 100;
        calclen();
    }
    Line(int x, int y, int x2, int y2) : Shape(x, y)
    {
        this->x2 = x2;
        this->y2 = y2;
        calclen();
    }
    void input()
    {
        Shape::input();
        cout << "Enter x2 and y2: ";
        cin >> x2 >> y2;
        calclen();
    }
    void output()
    {
        Shape::output();
        cout << "x2 = " << x2 << " y2 = " << y2 << endl;
        cout << "Len line = " << len << endl;
    }
    double getPer() const
    {
        return len;
    }
    void calclen()
    {
        len = sqrt((pow(x2 - x, 2) + pow(y2 - y, 2)));
    }
};

class Square : public Line
{
protected:
    int sidelen;

public:
    Square() : Line()
    {
        sidelen = rand() % 100;
    }
    Square(int x1, int y1, int x2, int y2) : Line(x1, y1, x2, y2)
    {
        sidelen = (sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2)));
    }
    void input()
    {
        Line::input();
        cout << "Enter side square: ";
        cin >> sidelen;
    }

    void output()
    {
        Line::output();
        cout << "Side square = " << sidelen << endl;
    }
    double getArea()
    {
        return sidelen * sidelen;
    }

    double getPer() const
    {
        return 4 * sidelen;
    }
};

class Rectangle : public Square
{
private:
    int width;
    int height;
    static unsigned int inp;
    static unsigned int outp;

public:
    Rectangle() : Square()
    {
        width = rand() % 100;
        height = rand() % 100;
        inp++;
        outp++;
    }
    Rectangle(int x1, int y1, int x2, int y2) : Square(x1, y1, x2, y2)
    {
        this->width = abs(x2 - x1);
        this->height = abs(y2 - y1);
        if (width == height)
        {
            cout << "don`t rectangle" << endl;
            exit(EXIT_FAILURE);
        }
        inp++;
        outp++;
    }
    void input()
    {
        Square::input();
        cout << "Enter width and height rectangle: ";
        cin >> width >> height;
    }
    void output()
    {
        Square::output();
        cout << "Width rectangle = " << width << endl;
        cout << "Height rectangle = " << height << endl;
        cout << "Per rectangle = " << getPer() << endl;
        cout << "Area rectangle = " << getArea() << endl;
    }

    double getArea()
    {
        return width * height;
    }

    double getPer() const
    {
        return 2 * (width + height);
    }
    static int getInp()
    {
        return inp;
    }

    static int getOutp()
    {
        return outp;
    }
    ~Rectangle()
    {
        outp--;
    }
};
unsigned int Rectangle ::inp = 0;
unsigned int Rectangle ::outp = 0;

class Circle : public Shape
{
private:
    double radius;

public:
    Circle() : Shape()
    {
        radius = rand() % 100;
    }

    Circle(int x, int y, double radius) : Shape(x, y)
    {
        this->radius = radius;
    }

    double getRadius() const
    {
        return radius;
    }
    double getPer() const
    {
        return 2 * 3.14159 * radius;
    }
};

class Ellipse : public Circle
{
private:
    double minorRadius;

public:
    Ellipse() : Circle()
    {
        minorRadius = rand() % 100;
    }

    Ellipse(int x, int y, double majorRadius, double minorRadius) : Circle(x, y, majorRadius)
    {
        this->minorRadius = minorRadius;
    }

    double getMinorRadius() const
    {
        return minorRadius;
    }

    double getMajorRadius() const
    {
        return getRadius();
    }

    double getArea() const
    {
        return 3.14159 * getMajorRadius() * minorRadius;
    }

    double getPerimeter() const
    {
        double h = ((getMajorRadius() - minorRadius) * (getMajorRadius() - minorRadius)) / ((getMajorRadius() + minorRadius) * (getMajorRadius() + minorRadius));
        return 3.14159 * (getMajorRadius() + minorRadius) * (1 + (3 * h) / (10 + sqrt(4 - 3 * h)));
    }
};
int main()
{
    Line line1(2, 4, 7, 3);
    Line line2;
    Rectangle rectangle1(8, 7, 9, 5);
    Rectangle rectangle2;
    Square square1(35, 21, 48, 28);
    Square square2;
    Ellipse ellipse1;
    Ellipse ellipse2(28, 41, 23, 84);
    Circle circle1;
    Circle circle2(28, 29, 6);

    cout << "Created object rectangle: " << Rectangle::getInp() << endl;
    cout << "Existing object rectangle: " << Rectangle::getOutp() << endl;

    Rectangle rectangle3;
    Rectangle *ptr = &rectangle1;

    cout << "Created object rectangle: " << Rectangle::getInp() << endl;
    cout << "Existing object rectangle:: " << Rectangle::getOutp() << endl;

    Rectangle *dynptr = new Rectangle(15, 8, 4, 6);

    cout << "Created object rectangle: " << Rectangle::getInp() << endl;
    cout << "Existing object rectangle:: " << Rectangle::getOutp() << endl;

    Shape *Shapes[4];
    Shapes[0] = new Square(2, 3, 8, 5);
    Shapes[1] = new Rectangle(9, 7, 3, 8);
    Shapes[2] = new Circle(5, 6, 3);
    Shapes[3] = new Ellipse(6, 3, 8, 5);

    cout << "Created object rectangle: " << Rectangle::getInp() << endl;
    cout << "Existing object rectangle:: " << Rectangle::getOutp() << endl;

    for (int i = 0; i < 4; i++)
    {
        cout << "per " << i + 1 << ": " << Shapes[i]->getPer() << endl;
    }

    for (int i = 0; i < 4; i++)
    {
        delete Shapes[i];
    }
    cout << "Created object rectangle: " << Rectangle::getInp() << endl;
    cout << "Existing object rectangle:: " << Rectangle::getOutp() << endl;

    delete dynptr;
    cout << "Created object rectangle: " << Rectangle::getInp() << endl;
    cout << "Existing object rectangle:: " << Rectangle::getOutp() << endl;
    return 0;
}
