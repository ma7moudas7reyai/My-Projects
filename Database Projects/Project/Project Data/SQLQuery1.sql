CREATE TABLE Models (
    Model_id INT PRIMARY KEY,
    Name VARCHAR(50) NOT NULL,
    Sex CHAR(1) CHECK (sex IN ('M', 'F')),
    Age INT,
    Address VARCHAR(255)
);

INSERT INTO Models (Model_id, Name, Sex, Age, Address) 
VALUES
(1, 'Ahmed Ali', 'M', 25, 'Cairo, Egypt'),
(2, 'Sarah Mohamed', 'F', 23, 'Alexandria, Egypt'),
(3, 'Mohamed Khaled', 'M', 30, 'Giza, Egypt'),
(4, 'Laila Nour', 'F', 27, 'Fayoum, Egypt');

CREATE TABLE Photograph (
    Photo_id INT PRIMARY KEY,
    Model_id INT,
    Photo_date DATE,
    FOREIGN KEY (Model_id) REFERENCES Models(Model_id) ON DELETE CASCADE
);


INSERT INTO Photograph (Photo_id, Model_id, Photo_date) VALUES
(101, 1, '2024-01-10'),
(102, 2, '2024-02-15'),
(103, 3, '2024-03-20'),
(104, 4, '2024-04-05');

CREATE TABLE Designers (
    Designer_id INT PRIMARY KEY,
    Name VARCHAR(50) NOT NULL,
    SSN VARCHAR(15) UNIQUE,
    Salary DECIMAL(10, 2),
    Address VARCHAR(255)
);


INSERT INTO Designers (Designer_id, Name, SSN, Salary, Address) VALUES
(201, 'Karim Hassan', '123-45-6789', 15000.50, 'Cairo, Egypt'),
(202, 'Amal Youssef', '987-65-4321', 17000.75, 'Alexandria, Egypt'),
(203, 'Youssef Rami', '456-78-9012', 16000.25, 'Giza, Egypt');

CREATE TABLE Products (
    Product_id INT PRIMARY KEY,
    Product_name VARCHAR(50) NOT NULL
);


INSERT INTO Products (Product_id, Product_name) VALUES
(501, 'Leather Shoes'),
(502, 'Sports Shoes'),
(503, 'Casual Shoes'),
(504, 'Formal Shoes');

CREATE TABLE Assistants (
    Assistant_id INT PRIMARY KEY,
    Name VARCHAR(50) NOT NULL,
    Sex CHAR(1) CHECK (sex IN ('M', 'F')),
    Designer_id INT,
    FOREIGN KEY (Designer_id) REFERENCES Designers(Designer_id) ON DELETE SET NULL
);

INSERT INTO Assistants (Assistant_id, Name, Sex) VALUES
(11, 'Heba Ali', 'F'),
(12, 'Omar Samir', 'M'),
(13, 'Nour Ashraf', 'F');

CREATE TABLE Departments (
    Department_id INT PRIMARY KEY,
    Name VARCHAR(50) NOT NULL,
    Locations VARCHAR(255)
);


INSERT INTO Departments (Department_id, Name, Locations) VALUES
(31, 'Design Department', 'Cairo, Egypt'),
(32, 'Marketing Department', 'Alexandria, Egypt'),
(33, 'Sales Department', 'Giza, Egypt');


CREATE TABLE Works_For (
    Designer_id INT,
    Department_id INT,
    Start_date DATE,
    PRIMARY KEY (Designer_id, Department_id),
    FOREIGN KEY (Designer_id) REFERENCES Designers(Designer_id) ON DELETE CASCADE,
    FOREIGN KEY (Department_id) REFERENCES Departments(Department_id) ON DELETE CASCADE
);


INSERT INTO Works_For (Designer_id, Department_id, Start_date) VALUES
(201, 31, '2024-01-01'),
(202, 32, '2024-02-01'),
(203, 33, '2024-03-01');


CREATE TABLE Designs (
    Designer_id INT,
    Product_id INT,
    PRIMARY KEY (Designer_id, Product_id),
    FOREIGN KEY (Designer_id) REFERENCES Designers(Designer_id) ON DELETE CASCADE,
    FOREIGN KEY (Product_id) REFERENCES Products(Product_id) ON DELETE CASCADE
);


INSERT INTO Designs (Designer_id, Product_id) VALUES
(201, 501),
(202, 502),
(203, 503);


CREATE TABLE Assist_Of (
    Assistant_id INT,
    Designer_id INT,
    Assigned_date DATE,
    PRIMARY KEY (Assistant_id, designer_id),
    FOREIGN KEY (Assistant_id) REFERENCES Assistants(Assistant_id) ON DELETE CASCADE,
    FOREIGN KEY (Designer_id) REFERENCES Designers(Designer_id) ON DELETE CASCADE
);


INSERT INTO Assist_Of (Assistant_id, Designer_id, Assigned_date) VALUES
(11, 201, '2024-01-10'),
(12, 202, '2024-02-15'),
(13, 203, '2024-03-20');



SELECT * FROM Models;

SELECT * FROM Photograph;

SELECT * FROM Designers;

SELECT * FROM Products;

SELECT * FROM Assistants;

SELECT * FROM Departments;

SELECT * FROM Works_For;

SELECT * FROM Designs;

SELECT * FROM Assist_Of;