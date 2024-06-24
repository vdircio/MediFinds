import React, { useState } from "react";
import "./SearchBar.css";

import { FaSearch } from 'react-icons/fa';

export const SearchBar = ({ setResults }) => {
    const [input, setInput] = useState("");
    const [loading, setLoading] = useState(false);
    const [marginTop, setMarginTop] = useState("2.5rem");

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            setMarginTop("-6.5rem");
            setLoading(true);
            search(input);
        }
    };

    const search = (query) => {
        fetch('http://127.0.0.1:5000/search', {  // Update the URL to point to your Flask server
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
        })
        .finally(() => {
            setLoading(false);
        })
    };

    return (
        <div className="orient" style={{marginTop: marginTop}}>
            <div>
                <h1>MediFinds</h1>
                <h3>A medical text summarization tool</h3>
                <h4>Needs to be optimized (wait around a minute / under progress)</h4>
            </div>
            <div className="input-wrapper">
                <div className="gcse-search"></div>
                <input 
                    placeholder="Type to search..." 
                    value={input} 
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                />
                <FaSearch id="search-icon"/>
            </div>
            {loading && <div className="loader"></div>}
        </div>
        
    );
};
