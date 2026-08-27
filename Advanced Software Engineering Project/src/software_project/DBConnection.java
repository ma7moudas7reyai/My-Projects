package software_project;

import java.sql.Connection;
import java.sql.DriverManager;

public class DBConnection {

    public static Connection connect() {
        try {
            Class.forName("com.microsoft.sqlserver.jdbc.SQLServerDriver");

            String url = System.getenv().getOrDefault(
                    "CAR_RENTAL_DB_URL",
                    "jdbc:sqlserver://localhost\\SQLEXPRESS;databaseName=Car_Rental_DB;encrypt=true;trustServerCertificate=true"
            );
            String user = System.getenv("CAR_RENTAL_DB_USER");
            String password = System.getenv("CAR_RENTAL_DB_PASSWORD");

            if (user == null || password == null) {
                throw new IllegalStateException(
                        "Set CAR_RENTAL_DB_USER and CAR_RENTAL_DB_PASSWORD before starting the application."
                );
            }

            return DriverManager.getConnection(url, user, password);

        } catch (Exception e) {
            throw new RuntimeException("Database connection failed", e);
        }
    }
}
