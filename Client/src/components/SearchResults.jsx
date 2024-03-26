import React from 'react';
import "./SearchResults.css";

export const SearchResults = ({ results }) => {
    return (
        <div className='results-list'>
            {Object.entries(results).map(([title, abstract], id) => (
                <div key={id}>
                    <h3>{title}</h3>
                    <p>{abstract}</p>
                </div>
            ))}
        </div>
    );
};
