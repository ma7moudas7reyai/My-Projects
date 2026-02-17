CREATE TABLE Patient 
(
    P_ID INT PRIMARY KEY,
    Name VARCHAR(100),
    Sex CHAR(1),
    Age INT,
    Address VARCHAR(255),
    City VARCHAR(100),
    State VARCHAR(100),
    PostalCode VARCHAR(20),
    Contact VARCHAR(20),
    P_Details TEXT
);

INSERT INTO Patient (P_ID, Name, Sex, Age, Address, City, State, PostalCode, Contact, P_Details) VALUES
(1, 'Ahmed Ali', 'M', 30, '123 Main St', 'Cairo', 'Cairo', '12345', '0123456789', 'Diabetic'),
(2, 'Sara Mohamed', 'F', 25, '456 Elm St', 'Giza', 'Giza', '67890', '0987654321', 'Asthmatic');





CREATE TABLE Doctor (
    Doc_ID INT PRIMARY KEY,
    Doc_Name VARCHAR(100),
    Address VARCHAR(255),
    Contact VARCHAR(20)
);

INSERT INTO Doctor (Doc_ID, Doc_Name, Address, Contact) VALUES
(1, 'Dr. Hany', '789 Pine St', '0112233445'),
(2, 'Dr. Salma', '101 Maple St', '0115566778');






CREATE TABLE Patient_Doctor (
    P_ID INT,
    Doc_ID INT,
    PRIMARY KEY (P_ID, Doc_ID),
    FOREIGN KEY (P_ID) REFERENCES Patient(P_ID),
    FOREIGN KEY (Doc_ID) REFERENCES Doctor(Doc_ID)
);

INSERT INTO Patient_Doctor (P_ID, Doc_ID) VALUES
(1, 1),
(2, 2);






CREATE TABLE Medicines (
    Drug_ID INT PRIMARY KEY,
    Manufacturer VARCHAR(100),
    BasePrice DECIMAL(10, 2),
    Tax DECIMAL(5, 2),
    Discount DECIMAL(5, 2),
    Price DECIMAL(10, 2),
    Mfg_Date DATE,
    Exp_Date DATE,
    Quantity INT
);

INSERT INTO Medicines (Drug_ID, Manufacturer, BasePrice, Tax, Discount, Price, Mfg_Date, Exp_Date, Quantity) VALUES
(1, 'PharmaCo', 100.00, 5.00, 10.00, 95.00, '2024-01-01', '2026-01-01', 50),
(2, 'MediHealth', 200.00, 10.00, 20.00, 180.00, '2023-06-01', '2025-06-01', 30);






CREATE TABLE Pharmacy (
    Pharmacy_ID INT PRIMARY KEY,
    Pharmacy_Name VARCHAR(100),
    Address VARCHAR(255),
    City VARCHAR(100),
    State VARCHAR(100),
    PostalCode VARCHAR(20),
    PIN_Code VARCHAR(20)
);

INSERT INTO Pharmacy (Pharmacy_ID, Pharmacy_Name, Address, City, State, PostalCode, PIN_Code) VALUES
(1, 'HealthPlus', '456 Pharmacy St', 'Cairo', 'Cairo', '54321', 'PIN123'),
(2, 'CareWell', '789 Medicine Rd', 'Giza', 'Giza', '98765', 'PIN456');






CREATE TABLE Pharmacy_Medicines (
    Pharmacy_ID INT,
    Drug_ID INT,
    PRIMARY KEY (Pharmacy_ID, Drug_ID),
    FOREIGN KEY (Pharmacy_ID) REFERENCES Pharmacy(Pharmacy_ID),
    FOREIGN KEY (Drug_ID) REFERENCES Medicines(Drug_ID)
);

INSERT INTO Pharmacy_Medicines (Pharmacy_ID, Drug_ID) VALUES
(1, 1),
(2, 2);





CREATE TABLE Supplier (
    Supp_ID INT PRIMARY KEY,
    Supp_Name VARCHAR(100),
    Location VARCHAR(100),
    Address VARCHAR(255),
    City VARCHAR(100),
    State VARCHAR(100),
    PostalCode VARCHAR(20),
    Quoted_Price DECIMAL(10, 2)
);

INSERT INTO Supplier (Supp_ID, Supp_Name, Location, Address, City, State, PostalCode, Quoted_Price) VALUES
(1, 'GlobalMed', 'Cairo', '123 Supplier Rd', 'Cairo', 'Cairo', '11223', 500.00),
(2, 'HealthSupplies', 'Giza', '456 Distributor Ave', 'Giza', 'Giza', '33445', 800.00);





CREATE TABLE Pharmacy_Supplier (
    Pharmacy_ID INT,
    Supp_ID INT,
    PRIMARY KEY (Pharmacy_ID, Supp_ID),
    FOREIGN KEY (Pharmacy_ID) REFERENCES Pharmacy(Pharmacy_ID),
    FOREIGN KEY (Supp_ID) REFERENCES Supplier(Supp_ID)
);

INSERT INTO Pharmacy_Supplier (Pharmacy_ID, Supp_ID) VALUES
(1, 1),
(2, 2);





CREATE TABLE Employee (
    Emp_ID INT PRIMARY KEY,
    Emp_Name VARCHAR(100),
    Salary DECIMAL(10, 2),
    Sex CHAR(1),
    Address VARCHAR(255),
    City VARCHAR(100),
    State VARCHAR(100),
    PostalCode VARCHAR(20),
    Pharmacy_ID INT,
    FOREIGN KEY (Pharmacy_ID) REFERENCES Pharmacy(Pharmacy_ID)
);

INSERT INTO Employee (Emp_ID, Emp_Name, Salary, Sex, ddress, City, State, PostalCode, Pharmacy_ID) VALUESA
(1, 'Ali Hassan', 5000.00, 'M', '12 Employee Ln', 'Cairo', 'Cairo', '65432', 1),
(2, 'Mona Ahmed', 4500.00, 'F', '34 Staff Rd', 'Giza', 'Giza', '87654', 2);


drop table Employee 




CREATE TABLE Bill (
    Bill_ID INT PRIMARY KEY,
    P_ID INT,
    Doc_ID INT,
    Drug_ID INT,
    AMT DECIMAL(10, 2),
    FOREIGN KEY (P_ID) REFERENCES Patient(P_ID),
    FOREIGN KEY (Doc_ID) REFERENCES Doctor(Doc_ID),
    FOREIGN KEY (Drug_ID) REFERENCES Medicines(Drug_ID)
);

INSERT INTO Bill (Bill_ID, P_ID, Doc_ID, Drug_ID, AMT) VALUES
(1, 1, 1, 1, 95.00),
(2, 2, 2, 2, 180.00);


SELECT * FROM Patient;

SELECT * FROM Doctor;

SELECT * FROM Patient_Doctor;

SELECT * FROM Medicines;

SELECT * FROM Pharmacy;

SELECT * FROM Pharmacy_Medicines;

SELECT * FROM Supplier;

SELECT * FROM Pharmacy_Supplier;

SELECT * FROM Employee;

SELECT * FROM Bill;
