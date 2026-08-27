# Car Rental Management System

A desktop car-rental application developed for the Advanced Software Engineering course at the Egyptian E-Learning University (EELU). It combines a Java Swing interface with a SQL Server database and separates customer and administrator workflows.

## Features

- User registration and login with input validation
- Role-based navigation for customers and administrators
- Browse available cars and record rent or purchase transactions
- Add and remove cars through the administrator dashboard
- View transaction history
- Prepared SQL statements for database operations
- JUnit tests for email and password validation

## Technologies

- Java 17 and Swing
- Microsoft SQL Server and JDBC
- Maven
- JUnit 4

## Project Structure

```text
src/software_project/       Application source and UI resources
test/software_project/      Validation tests
database/schema.sql         Database schema and sample data
pom.xml                     Maven build configuration
```

## Run Locally

1. Create a SQL Server database named `Car_Rental_DB`.
2. Review and run `database/schema.sql` using SQL Server Management Studio.
3. Set the database credentials in environment variables:

```powershell
$env:CAR_RENTAL_DB_USER = "your-user"
$env:CAR_RENTAL_DB_PASSWORD = "your-password"
```

Optionally set `CAR_RENTAL_DB_URL` if SQL Server is not running on the default local `SQLEXPRESS` instance.

4. Build and test the project:

```bash
mvn clean test
```

5. Start the application:

```bash
mvn exec:java
```

## Notes

This repository contains the source code and a reproducible SQL script. Local database files, compiled classes, IDE metadata, and credentials are intentionally excluded.
