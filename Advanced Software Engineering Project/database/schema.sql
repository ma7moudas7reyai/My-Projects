USE Car_Rental_DB;
GO

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name = 'Users' AND xtype = 'U')
CREATE TABLE Users (
    id INT PRIMARY KEY IDENTITY(1, 1),
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(100) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user'
);

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name = 'Cars' AND xtype = 'U')
CREATE TABLE Cars (
    id INT PRIMARY KEY IDENTITY(1, 1),
    name VARCHAR(100) NOT NULL,
    model VARCHAR(50),
    price DECIMAL(10,2),
    status VARCHAR(20) DEFAULT 'available'
);

IF NOT EXISTS (SELECT * FROM sys.check_constraints WHERE name = 'CHK_Status')
BEGIN
    ALTER TABLE Cars
    ADD CONSTRAINT CHK_Status
    CHECK (status IN ('available', 'rented', 'sold'));
END

IF NOT EXISTS (SELECT * FROM sysobjects WHERE name = 'Transactions' AND xtype = 'U')
CREATE TABLE Transactions (
    id INT PRIMARY KEY IDENTITY(1, 1),
    user_id INT NULL,
    admin_id INT NULL,
    car_id INT,
    car_name VARCHAR(100),
    type VARCHAR(20) NOT NULL,
    date DATETIME DEFAULT GETDATE(),

    FOREIGN KEY (user_id) REFERENCES Users(id),
    FOREIGN KEY (admin_id) REFERENCES Users(id),
    FOREIGN KEY (car_id) REFERENCES Cars(id)
);

IF NOT EXISTS (SELECT * FROM sys.check_constraints WHERE name = 'CHK_Type')
BEGIN
    ALTER TABLE Transactions
    ADD CONSTRAINT CHK_Type 
    CHECK (type IN ('rent', 'buy', 'add', 'delete'));
END

IF NOT EXISTS (SELECT * FROM sys.check_constraints WHERE name = 'CHK_UserOrAdmin')
BEGIN
    ALTER TABLE Transactions
    ADD CONSTRAINT CHK_UserOrAdmin
    CHECK (
        (user_id IS NOT NULL AND admin_id IS NULL)
        OR
        (user_id IS NULL AND admin_id IS NOT NULL)
    );
END

IF NOT EXISTS (SELECT * FROM Users WHERE role = 'admin')
BEGIN
    INSERT INTO Users (name, email, password, role)
    VALUES ('Admin', 'admin@example.com', 'change-me-before-use', 'admin');
END

IF NOT EXISTS (SELECT * FROM Cars)
INSERT INTO Cars (name, model, price, status)
VALUES 
('BMW', 'X6', 50000, 'available'),
('Audi', 'A7', 60000, 'available'),
('Mercedes', 'C200', 55000, 'available'),
('Toyota', 'Corolla', 20000, 'available'),
('Honda', 'Civic', 22000, 'available'),
('Ford', 'Mustang', 45000, 'available');

SELECT 
    t.car_name,
    t.type,

    CASE 
        WHEN t.user_id IS NOT NULL THEN u.name
        ELSE a.name
    END AS done_by,

    CASE 
        WHEN t.user_id IS NOT NULL THEN 'User'
        ELSE 'Admin'
    END AS role,

    t.date

FROM Transactions t
LEFT JOIN Users u ON t.user_id = u.id
LEFT JOIN Users a ON t.admin_id = a.id

ORDER BY t.date DESC;

SELECT * FROM Cars;
SELECT * FROM Users;
SELECT * FROM Transactions;
