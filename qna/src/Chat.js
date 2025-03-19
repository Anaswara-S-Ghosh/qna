import React, { useState } from "react";
import "./Chat.css";
import axios from "axios";

function Chat() {
  const [userInput, setUserInput] = useState("");
  const [chatHistory, setChatHistory] = useState([]);

  const sendQuery = async (query) => {
    if (!query.trim()) return;
    setChatHistory([...chatHistory, { sender: "user", message: query }]);
    setUserInput("");

    try {
      const res = await axios.post("http://localhost:5000/user_query", { 
          question: query  // Changed from 'query' to 'question'
      });
      
      if (res) {
        console.log(res)
        setChatHistory((prev) => [
          ...prev,
          { sender: "bot", message: res.data.relevant_chunks },
        ]);
      }
    } catch (error) {
      console.error("Error sending query:", error);
      setChatHistory((prev) => [
        ...prev,
        { sender: "bot", message: "Sorry, something went wrong." },
      ]);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        {/* Sidebar content */}
        <div className="sidebar">
          <div className="menu">
           <div className="menu-item active">
             <img src="../icons/chats.png" alt="Chat" className="menu-icon" />
             Chat
            </div>
          <div className="menu-item">
            <img src="../icons/chathistory.png" alt="History" className="menu-icon" />
            Chat History
          </div>
          <div className="menu-item">
            <img src="../icons/aiassistant.png" alt="Personalities" className="menu-icon" />
            AI Personalities
          </div>
          <div className="menu-item">
            <img src="../icons/setting.png" alt="Settings" className="menu-icon" />
            Settings
          </div>
        </div>
        <div className="upgrade-button">
          <button>Upgrade to Pro</button>
          <p>Unlock powerful features with our pro upgrade today!</p>
        </div>
      </div>
      </div>

      {/* Main Content Area */}
      <div className="main-content">
        <div className="chat-window">
          {chatHistory.map((chat, index) => (
            <div
              key={index}
              className={`chat-message ${chat.sender === "user" ? "user" : "bot"}`}
            >
              {
                chat.message
              }
            </div>
          ))}
        </div>

        <div className="query-box">
          <input
            type="text"
            placeholder="Enter your query here"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            onKeyPress={(e) => {
              if (e.key === "Enter") sendQuery(userInput);
            }}
          />
          <button onClick={() => sendQuery(userInput)}>Send</button>
        </div>
      </div>
    </div>
  );
}

export default Chat;
