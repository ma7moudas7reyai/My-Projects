package software_project;

import java.awt.Color;
import java.sql.Connection;
import java.sql.PreparedStatement;
import javax.swing.JOptionPane;
import java.sql.ResultSet;
import javax.swing.ImageIcon;

public class SignUp extends javax.swing.JFrame {
    
    private static final java.util.logging.Logger logger = java.util.logging.Logger.getLogger(SignUp.class.getName());

    public SignUp() {
        initComponents();
        logo2.setIcon(new ImageIcon(getClass().getResource("/software_project/icons/ChatGPT-Image-Apr-27-2026-10-48.png")));
        Color normalColor = new Color(225,29,72);
        Color hoverColor = new Color(170,10,50);

        signUpBTN2.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseEntered(java.awt.event.MouseEvent evt) {
                signUpBTN2.setBackground(hoverColor);
            }

            public void mouseExited(java.awt.event.MouseEvent evt) {
                signUpBTN2.setBackground(normalColor);
            }
        });
        fullNameText.setCaretColor(Color.WHITE);
        emailText2.setCaretColor(Color.WHITE);
        passwordText2.setCaretColor(Color.WHITE);
        confirmPasswordText.setCaretColor(Color.WHITE);
    }

    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        left2 = new javax.swing.JPanel();
        logo2 = new javax.swing.JLabel();
        comapnyName2 = new javax.swing.JLabel();
        right2 = new javax.swing.JPanel();
        signUpLogo = new javax.swing.JLabel();
        fullName = new javax.swing.JLabel();
        fullNameText = new javax.swing.JTextField();
        email2 = new javax.swing.JLabel();
        password2 = new javax.swing.JLabel();
        confirmPassword = new javax.swing.JLabel();
        emailText2 = new javax.swing.JTextField();
        passwordText2 = new javax.swing.JPasswordField();
        confirmPasswordText = new javax.swing.JPasswordField();
        signUpBTN2 = new javax.swing.JButton();
        note2 = new javax.swing.JLabel();
        loginBTN2 = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Sign Up");
        setPreferredSize(new java.awt.Dimension(800, 500));

        jPanel1.setBackground(new java.awt.Color(255, 255, 255));
        jPanel1.setPreferredSize(new java.awt.Dimension(800, 500));
        jPanel1.setLayout(null);

        left2.setBackground(new java.awt.Color(15, 23, 42));

        comapnyName2.setFont(new java.awt.Font("Segoe Script", 1, 36)); // NOI18N
        comapnyName2.setForeground(new java.awt.Color(167, 243, 208));
        comapnyName2.setText("Car Rental");

        javax.swing.GroupLayout left2Layout = new javax.swing.GroupLayout(left2);
        left2.setLayout(left2Layout);
        left2Layout.setHorizontalGroup(
            left2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(left2Layout.createSequentialGroup()
                .addGap(92, 92, 92)
                .addGroup(left2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(logo2, javax.swing.GroupLayout.PREFERRED_SIZE, 220, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(comapnyName2))
                .addContainerGap(88, Short.MAX_VALUE))
        );
        left2Layout.setVerticalGroup(
            left2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(left2Layout.createSequentialGroup()
                .addGap(103, 103, 103)
                .addComponent(logo2, javax.swing.GroupLayout.PREFERRED_SIZE, 166, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(comapnyName2)
                .addContainerGap(166, Short.MAX_VALUE))
        );

        jPanel1.add(left2);
        left2.setBounds(0, 0, 400, 500);

        right2.setBackground(new java.awt.Color(30, 41, 59));

        signUpLogo.setFont(new java.awt.Font("Segoe UI", 1, 36)); // NOI18N
        signUpLogo.setForeground(new java.awt.Color(255, 255, 255));
        signUpLogo.setText("Sign Up");

        fullName.setBackground(new java.awt.Color(51, 65, 85));
        fullName.setFont(new java.awt.Font("Segoe UI", 0, 16)); // NOI18N
        fullName.setForeground(new java.awt.Color(203, 213, 225));
        fullName.setText("Full Name");

        fullNameText.setBackground(new java.awt.Color(15, 23, 42));
        fullNameText.setFont(new java.awt.Font("Segoe UI", 0, 16)); // NOI18N
        fullNameText.setForeground(new java.awt.Color(255, 255, 255));

        email2.setBackground(new java.awt.Color(51, 65, 85));
        email2.setFont(new java.awt.Font("Segoe UI", 0, 16)); // NOI18N
        email2.setForeground(new java.awt.Color(203, 213, 225));
        email2.setText("Email");

        password2.setBackground(new java.awt.Color(51, 65, 85));
        password2.setFont(new java.awt.Font("Segoe UI", 0, 16)); // NOI18N
        password2.setForeground(new java.awt.Color(203, 213, 225));
        password2.setText("Password");

        confirmPassword.setBackground(new java.awt.Color(51, 65, 85));
        confirmPassword.setFont(new java.awt.Font("Segoe UI", 0, 16)); // NOI18N
        confirmPassword.setForeground(new java.awt.Color(203, 213, 225));
        confirmPassword.setText("Confirm Password");

        emailText2.setBackground(new java.awt.Color(15, 23, 42));
        emailText2.setFont(new java.awt.Font("Segoe UI", 0, 16)); // NOI18N
        emailText2.setForeground(new java.awt.Color(255, 255, 255));
        emailText2.addActionListener(this::emailText2ActionPerformed);

        passwordText2.setBackground(new java.awt.Color(15, 23, 42));
        passwordText2.setForeground(new java.awt.Color(255, 255, 255));

        confirmPasswordText.setBackground(new java.awt.Color(15, 23, 42));
        confirmPasswordText.setForeground(new java.awt.Color(255, 255, 255));

        signUpBTN2.setBackground(new java.awt.Color(225, 29, 72));
        signUpBTN2.setForeground(new java.awt.Color(255, 255, 255));
        signUpBTN2.setText("Sign Up");
        signUpBTN2.addActionListener(this::signUpBTN2ActionPerformed);

        note2.setForeground(new java.awt.Color(148, 163, 184));
        note2.setText("I don't have an account");

        loginBTN2.setBackground(new java.awt.Color(225, 29, 72));
        loginBTN2.setForeground(new java.awt.Color(255, 255, 255));
        loginBTN2.setText("Login");
        loginBTN2.addActionListener(this::loginBTN2ActionPerformed);

        javax.swing.GroupLayout right2Layout = new javax.swing.GroupLayout(right2);
        right2.setLayout(right2Layout);
        right2Layout.setHorizontalGroup(
            right2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(right2Layout.createSequentialGroup()
                .addGroup(right2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(right2Layout.createSequentialGroup()
                        .addGap(121, 121, 121)
                        .addComponent(signUpLogo))
                    .addComponent(fullName)
                    .addGroup(right2Layout.createSequentialGroup()
                        .addContainerGap()
                        .addComponent(fullNameText, javax.swing.GroupLayout.PREFERRED_SIZE, 350, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addGroup(right2Layout.createSequentialGroup()
                        .addContainerGap()
                        .addComponent(emailText2, javax.swing.GroupLayout.PREFERRED_SIZE, 350, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addComponent(password2)
                    .addComponent(email2)
                    .addComponent(confirmPassword)
                    .addGroup(right2Layout.createSequentialGroup()
                        .addContainerGap()
                        .addComponent(passwordText2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addGroup(right2Layout.createSequentialGroup()
                        .addContainerGap()
                        .addComponent(confirmPasswordText, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addGroup(right2Layout.createSequentialGroup()
                        .addContainerGap()
                        .addGroup(right2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(signUpBTN2, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(note2)))
                    .addGroup(right2Layout.createSequentialGroup()
                        .addContainerGap()
                        .addComponent(loginBTN2, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addContainerGap(44, Short.MAX_VALUE))
        );

        right2Layout.linkSize(javax.swing.SwingConstants.HORIZONTAL, new java.awt.Component[] {confirmPasswordText, emailText2, passwordText2});

        right2Layout.setVerticalGroup(
            right2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(right2Layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(signUpLogo)
                .addGap(3, 3, 3)
                .addComponent(fullName, javax.swing.GroupLayout.PREFERRED_SIZE, 24, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(fullNameText, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(email2, javax.swing.GroupLayout.PREFERRED_SIZE, 24, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(emailText2, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(password2, javax.swing.GroupLayout.PREFERRED_SIZE, 24, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(passwordText2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(confirmPassword, javax.swing.GroupLayout.PREFERRED_SIZE, 24, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(confirmPasswordText, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(signUpBTN2, javax.swing.GroupLayout.PREFERRED_SIZE, 39, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(note2)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(loginBTN2, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(40, Short.MAX_VALUE))
        );

        right2Layout.linkSize(javax.swing.SwingConstants.VERTICAL, new java.awt.Component[] {confirmPasswordText, emailText2, passwordText2});

        jPanel1.add(right2);
        right2.setBounds(400, 0, 400, 500);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, 503, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    public static boolean isValidEmail(String email) {
        String regex = "^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$";
        return email.matches(regex);
    }
    
    public static boolean isValidName(String name) {
        name = name.trim();

        if(name.isEmpty()) return false;

        if(!name.matches("^[A-Za-z0-9 ]+$")) return false;

        if(!name.matches(".*[A-Za-z].*")) return false;

        return true;
    }
    
    public static boolean isValidPassword(String password) {
        return password.matches("^[^\\s]{8,}$");
    }
    
    private void emailText2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_emailText2ActionPerformed
        // TODO add your handling code here:
    }//GEN-LAST:event_emailText2ActionPerformed

    private void signUpBTN2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_signUpBTN2ActionPerformed
        String name = fullNameText.getText();
        String email = emailText2.getText();
        String password = new String(passwordText2.getPassword());
        String confirmPassword = new String(confirmPasswordText.getPassword());

        if(!isValidName(name)) {
            JOptionPane.showMessageDialog(this, "Name must contain letters and can include numbers and spaces (not numbers only)");
            fullNameText.requestFocus();
            return;
        }

        if(!isValidEmail(email)) {
            JOptionPane.showMessageDialog(this, "Invalid Email Format");
            emailText2.requestFocus();
            return;
        }

        if(!isValidPassword(password)) {
            JOptionPane.showMessageDialog(this, "Password must be at least 8 characters and contain no spaces");
            passwordText2.requestFocus();
            return;
        }

        if(!password.equals(confirmPassword)) {
            JOptionPane.showMessageDialog(this, "Passwords do not match");
            confirmPasswordText.requestFocus();
            return;
        }

        Connection con = null;
        
        try {
            con = DBConnection.connect();
            String query = "INSERT INTO Users (name, email, password, role) VALUES (?, ?, ?, ?)";
            PreparedStatement pst = con.prepareStatement(query);
            pst.setString(1, name);
            pst.setString(2, email);
            pst.setString(3, password);
            pst.setString(4, "user");
            pst.executeUpdate();
            JOptionPane.showMessageDialog(this, "Account Created Successfully");
            String getIdQuery = "SELECT id FROM Users WHERE email = ?";
            PreparedStatement getIdPst = con.prepareStatement(getIdQuery);
            getIdPst.setString(1, email);
            ResultSet rs = getIdPst.executeQuery();

            int userId = 0;
            if(rs.next()) {
                userId = rs.getInt("id");
            }

            HomePage home = new HomePage(userId);
            home.pack();
            home.setLocationRelativeTo(null);
            home.setVisible(true);
            this.dispose();

        } catch (Exception e) {
            if(e.getMessage().contains("UNIQUE")) {
                JOptionPane.showMessageDialog(this, "Email already exists");
            } else {
                JOptionPane.showMessageDialog(this, "Error: " + e.getMessage());
            }
        } finally {
            try {
                if(con != null) con.close();
            } catch(Exception e){}
        }
    }//GEN-LAST:event_signUpBTN2ActionPerformed

    private void loginBTN2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_loginBTN2ActionPerformed
        Login LoginFrame = new Login();
        LoginFrame.pack();
        LoginFrame.setLocationRelativeTo(null);
        LoginFrame.setVisible(true);
        this.dispose();
    }//GEN-LAST:event_loginBTN2ActionPerformed

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel comapnyName2;
    private javax.swing.JLabel confirmPassword;
    private javax.swing.JPasswordField confirmPasswordText;
    private javax.swing.JLabel email2;
    private javax.swing.JTextField emailText2;
    private javax.swing.JLabel fullName;
    private javax.swing.JTextField fullNameText;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel left2;
    private javax.swing.JButton loginBTN2;
    private javax.swing.JLabel logo2;
    private javax.swing.JLabel note2;
    private javax.swing.JLabel password2;
    private javax.swing.JPasswordField passwordText2;
    private javax.swing.JPanel right2;
    private javax.swing.JButton signUpBTN2;
    private javax.swing.JLabel signUpLogo;
    // End of variables declaration//GEN-END:variables
}
