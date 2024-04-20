import React, { useState } from "react";
import "./SearchBar.css";

import { FaSearch } from 'react-icons/fa';

export const SearchBar = ({ setResults }) => {
    const [input, setInput] = useState("");

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            search(input);
        }
    };

    const search = (query) => {
        fetch('http://209.38.130.148:5000/search', {  // Update the URL to point to your Flask server
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        })
        .then(response => response.json())
        .then(data => {
            setResults(data);
        })
        .catch(error => {
            console.error('Error searching:', error);
        });
    };

    return (
        <div className="orient">
            <div>
                <h1>MediFinds</h1>
                <h2>A medical text summarization tool</h2>
                <h3>Needs to be optimized (wait around a minute / under progress)</h3>
            </div>
            <div className="input-wrapper">
                <FaSearch id="search-icon"/>
                <div className="gcse-search"></div>
                <input 
                    placeholder="Type to search..." 
                    value={input} 
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                />
            </div>
        </div>
        
    );
};
