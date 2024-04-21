import React from 'react';
import "./SearchResults.css";

export const SearchResults = ({ results }) => {
    return (
        <div className='results-list'>
            {Object.map(({link, title, abstract}, id) => (
                <div key={id}>
                    <a href = {link} target="_blank" rel="noopener noreferrer"> 
                        <h3>{title}</h3>
                        <p>{abstract}</p>
                    </a>
                </div>
            ))}
        </div>
    );
};
