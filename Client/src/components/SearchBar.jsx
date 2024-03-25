import React, {useState} from "react";
import "./SearchBar.css";

import {FaSearch} from 'react-icons/fa'

export const SearchBar = () => {
    const [input, setInput] = useState("");

    const fetchData = (value) => {
        
    }

    return <div className="input-wrapper">
        <FaSearch id="search-icon"/>
        <script async src="https://cse.google.com/cse.js?cx=7312e2f7473b445d3"></script>
        <div class="gcse-search"></div>
        <input placeholder="Type to search..." value={input} onChange={(e) => setInput(e.target.value)}/>
    </div>;
}