package software_project;

import java.awt.Color;
import java.util.regex.Pattern;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import javax.swing.ImageIcon;
import javax.swing.JOptionPane;

public class Login extends javax.swing.JFrame {

    private static final java.util.logging.Logger logger = java.util.logging.Logger.getLogger(Login.class.getName());

    public Login() {
        initComponents();
        logo.setIcon(new ImageIcon(getClass().getResource("/software_project/icons/ChatGPT-Image-Apr-27-2026-10-48.png")));

        emailText.setCaretColor(Color.WHITE);
        passwordText.setCaretColor(Color.WHITE);
        passwordText.setPreferredSize(new java.awt.Dimension(350, 35));

        loginBTN.addMouseListener(new java.awt.event.MouseAdapter() {
            public void mouseEntered(java.awt.event.MouseEvent evt) {
                loginBTN.setBackground(new Color(170,10,50));
            }

            public void mouseExited(java.awt.event.MouseEvent evt) {
                loginBTN.setBackground(new Color(225,29,72));
            }
        });
        
    }

    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel2 = new javax.swing.JPanel();
        Left = new javax.swing.JPanel();
        logo = new javax.swing.JLabel();
        comapnyName = new javax.swing.JLabel();
        Right = new javax.swing.JPanel();
        loginLogo = new javax.swing.JLabel();
        email = new javax.swing.JLabel();
        emailText = new javax.swing.JTextField();
        password = new javax.swing.JLabel();
        passwordText = new javax.swing.JPasswordField();
        loginBTN = new javax.swing.JButton();
        note = new javax.swing.JLabel();
        signUpBTN = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Login");

        jPanel2.setBackground(new java.awt.Color(255, 255, 255));
        jPanel2.setPreferredSize(new java.awt.Dimension(800, 500));
        jPanel2.setLayout(null);

        Left.setBackground(new java.awt.Color(15, 23, 42));
        Left.setPreferredSize(new java.awt.Dimension(400, 500));

        logo.setIcon(new javax.swing.ImageIcon(getClass().getResource("/software_project/icons/ChatGPT-Image-Apr-27-2026-10-48.png"))); // NOI18N

        comapnyName.setBackground(new java.awt.Color(167, 243, 208));
        comapnyName.setFont(new java.awt.Font("Segoe Script", 1, 36)); // NOI18N
        comapnyName.setForeground(new java.awt.Color(176, 228, 204));
        comapnyName.setText("Car Rental");

        javax.swing.GroupLayout LeftLayout = new javax.swing.GroupLayout(Left);
        Left.setLayout(LeftLayout);
        LeftLayout.setHorizontalGroup(
            LeftLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(LeftLayout.createSequentialGroup()
                .addGap(92, 92, 92)
                .addGroup(LeftLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(logo, javax.swing.GroupLayout.PREFERRED_SIZE, 220, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(comapnyName))
                .addContainerGap(88, Short.MAX_VALUE))
        );
        LeftLayout.setVerticalGroup(
            LeftLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(LeftLayout.createSequentialGroup()
                .addGap(91, 91, 91)
                .addComponent(logo, javax.swing.GroupLayout.PREFERRED_SIZE, 166, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(comapnyName)
                .addContainerGap(166, Short.MAX_VALUE))
        );

        jPanel2.add(Left);
        Left.setBounds(0, 0, 400, 500);
        Left.getAccessibleContext().setAccessibleDescription("");

        Right.setBackground(new java.awt.Color(30, 41, 59));
        Right.setPreferredSize(new java.awt.Dimension(400, 500));

        loginLogo.setBackground(new java.awt.Color(255, 255, 255));
        loginLogo.setFont(new java.awt.Font("Segoe UI", 1, 36)); // NOI18N
        loginLogo.setForeground(new java.awt.Color(255, 255, 255));
        loginLogo.setText("Login");

        email.setBackground(new java.awt.Color(176, 228, 204));
        email.setFont(new java.awt.Font("Segoe UI", 0, 16)); // NOI18N
        email.setForeground(new java.awt.Color(203, 213, 225));
        email.setText("Email");

        emailText.setBackground(new java.awt.Color(15, 23, 42));
        emailText.setFont(new java.awt.Font("Segoe UI", 0, 16)); // NOI18N
        emailText.setForeground(new java.awt.Color(255, 255, 255));
        emailText.setBorder(javax.swing.BorderFactory.createTitledBorder(""));
        emailText.addActionListener(this::emailTextActionPerformed);

        password.setBackground(new java.awt.Color(176, 228, 204));
        password.setFont(new java.awt.Font("Segoe UI", 0, 16)); // NOI18N
        password.setForeground(new java.awt.Color(203, 213, 225));
        password.setText("Password");

        passwordText.setBackground(new java.awt.Color(15, 23, 42));
        passwordText.setForeground(new java.awt.Color(255, 255, 255));
        passwordText.setBorder(javax.swing.BorderFactory.createTitledBorder(""));
        passwordText.addActionListener(this::passwordTextActionPerformed);

        loginBTN.setBackground(new java.awt.Color(225, 29, 72));
        loginBTN.setForeground(new java.awt.Color(255, 255, 255));
        loginBTN.setText("Login");
        loginBTN.addActionListener(this::loginBTNActionPerformed);

        note.setBackground(new java.awt.Color(148, 163, 184));
        note.setForeground(new java.awt.Color(148, 163, 184));
        note.setText("I don't have an account");

        signUpBTN.setBackground(new java.awt.Color(225, 29, 72));
        signUpBTN.setForeground(new java.awt.Color(255, 255, 255));
        signUpBTN.setText("Sign Up");
        signUpBTN.addActionListener(this::signUpBTNActionPerformed);

        javax.swing.GroupLayout RightLayout = new javax.swing.GroupLayout(Right);
        Right.setLayout(RightLayout);
        RightLayout.setHorizontalGroup(
            RightLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(RightLayout.createSequentialGroup()
                .addGroup(RightLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(RightLayout.createSequentialGroup()
                        .addContainerGap()
                        .addGroup(RightLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(email, javax.swing.GroupLayout.PREFERRED_SIZE, 74, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(emailText, javax.swing.GroupLayout.PREFERRED_SIZE, 350, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(password, javax.swing.GroupLayout.PREFERRED_SIZE, 74, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(passwordText, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addComponent(note)
                            .addComponent(loginBTN, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)))
                    .addGroup(RightLayout.createSequentialGroup()
                        .addGap(140, 140, 140)
                        .addComponent(loginLogo))
                    .addGroup(RightLayout.createSequentialGroup()
                        .addContainerGap()
                        .addComponent(signUpBTN, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addContainerGap(44, Short.MAX_VALUE))
        );

        RightLayout.linkSize(javax.swing.SwingConstants.HORIZONTAL, new java.awt.Component[] {emailText, passwordText});

        RightLayout.setVerticalGroup(
            RightLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(RightLayout.createSequentialGroup()
                .addGap(42, 42, 42)
                .addComponent(loginLogo)
                .addGap(60, 60, 60)
                .addComponent(email, javax.swing.GroupLayout.PREFERRED_SIZE, 24, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(emailText, javax.swing.GroupLayout.PREFERRED_SIZE, 35, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(password, javax.swing.GroupLayout.PREFERRED_SIZE, 24, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(passwordText, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(18, 18, 18)
                .addComponent(loginBTN, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(note)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(signUpBTN, javax.swing.GroupLayout.PREFERRED_SIZE, 40, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(76, Short.MAX_VALUE))
        );

        RightLayout.linkSize(javax.swing.SwingConstants.VERTICAL, new java.awt.Component[] {emailText, passwordText});

        jPanel2.add(Right);
        Right.setBounds(400, 0, 400, 500);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanel2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    public static boolean isValidEmail(String email) {
        String regex = "^[A-Za-z0-9+_.-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$";
        return Pattern.matches(regex, email);
    }
    
    public static boolean isValidPassword(String password) {
        String regex = "^[^\\s]{8,}$";
        return password.matches(regex);
    }
    
    private void signUpBTNActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_signUpBTNActionPerformed
        SignUp SignUpFrame = new SignUp();
        SignUpFrame.pack();
        SignUpFrame.setLocationRelativeTo(null);
        SignUpFrame.setVisible(true);
        this.dispose();
    }//GEN-LAST:event_signUpBTNActionPerformed

    private void emailTextActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_emailTextActionPerformed
       
    }//GEN-LAST:event_emailTextActionPerformed

    private void passwordTextActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_passwordTextActionPerformed
        
    }//GEN-LAST:event_passwordTextActionPerformed

    private void loginBTNActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_loginBTNActionPerformed

        String email = emailText.getText().trim();
        String password = new String(passwordText.getPassword()).trim();

        if(email.isEmpty() || password.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Please fill in all fields");
            return;
        }

        if(!isValidEmail(email)) {
            JOptionPane.showMessageDialog(this, "Invalid Email Format");
            return;
        }

        if(!isValidPassword(password)) {
            JOptionPane.showMessageDialog(this, "Password must be at least 8 characters and contain no spaces");
            return;
        }

        loginBTN.setEnabled(false);
        Connection con = null;

        try {
            con = DBConnection.connect();
            String query = "SELECT id, role FROM Users WHERE email = ? AND password = ?";
            PreparedStatement pst = con.prepareStatement(query);
            pst.setString(1, email);
            pst.setString(2, password);
            ResultSet rs = pst.executeQuery();

            if(rs.next()) {
                int userId = rs.getInt("id"); 
                String role = rs.getString("role");

                if(role.equals("admin")) {
                    Admins admin = new Admins(userId);
                    admin.pack();
                    admin.setLocationRelativeTo(null);
                    admin.setVisible(true);

                } else {
                    HomePage home = new HomePage(userId);
                    home.pack();
                    home.setLocationRelativeTo(null);
                    home.setVisible(true);
                }

                this.dispose(); 

            } else {
                JOptionPane.showMessageDialog(this, "Invalid Email or Password!");
            }

        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Error: " + e.getMessage());
        } finally {
            try {
                if(con != null) con.close();
            } catch(Exception e){}
            loginBTN.setEnabled(true);
        }
    }//GEN-LAST:event_loginBTNActionPerformed

    public static void main(String args[]) {
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ReflectiveOperationException | javax.swing.UnsupportedLookAndFeelException ex) {
            logger.log(java.util.logging.Level.SEVERE, null, ex);
        }
        java.awt.EventQueue.invokeLater(() -> new Login().setVisible(true));
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JPanel Left;
    private javax.swing.JPanel Right;
    private javax.swing.JLabel comapnyName;
    private javax.swing.JLabel email;
    private javax.swing.JTextField emailText;
    private javax.swing.JPanel jPanel2;
    private javax.swing.JButton loginBTN;
    private javax.swing.JLabel loginLogo;
    private javax.swing.JLabel logo;
    private javax.swing.JLabel note;
    private javax.swing.JLabel password;
    private javax.swing.JPasswordField passwordText;
    private javax.swing.JButton signUpBTN;
    // End of variables declaration//GEN-END:variables
}
