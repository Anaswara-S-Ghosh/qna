import React from "react";
import "./App.css"; // CSS file for styling
import { Link } from "react-router-dom";

function LoginPage() {
  return (
    <div className="login-container">
      <div className="login-form">
        <div className="logo">
          <img src="/logo.png" alt="LegalLens " className="logo" />
          <h1>LegalLens</h1>
        </div>
        <h2>Login</h2>
        <form>
          <div className="form-group">
            <label htmlFor="email">Your email*</label>
            <input
              type="email"
              id="email"
              placeholder="Enter your email"
              required
              pattern="[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
              title="Enter a valid email address"
            />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password*</label>
            <input
              type="password"
              id="password"
              placeholder="Enter your password"
              required
            />
            <button type="button" className="toggle-password">
              👁
            </button>
          </div>
          <div className="form-group">
            <input type="checkbox" id="terms" required />
            <label htmlFor="terms">I agree to terms & conditions</label>
          </div>

          <Link to="/chat">
          <button className="btn primary-btn">
              Sign in
          </button>
          </Link>


        </form>
        <div className="separator">or</div>
        <button className="btn google-btn">Login with Google</button>
        {/* <Link to="/signup"> */}
        <p className="signup-link">
          Don't have an Account? <a href="/signup">Create an account</a>
        </p>
 
        {/* Forgot Password Link */}
        <p className="forgot-password-link">
          <a href="/forgot-password">Forgot Password?</a>
        </p>
      </div>
      <div className="login-image">
        <img
          src="https://bharatchugh.in/wp-content/uploads/2020/12/pexels-photo-5669602.jpeg?w=823" // Replace with actual scales image
          alt="Scales of Justice"
        />
      </div>
    </div>
  );
}

export default LoginPage;
