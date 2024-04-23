import React from 'react';
import "./SearchResults.css";

export const SearchResults = ({ results }) => {
    if (results.length > 0) {
        results = JSON.parse(results)
    }

    console.log(results)

    return (
        <div className='results-list'>
            {Object.keys(results).map((id) => (
                <div key={id}>
                    <a href={results[id]["link"]} target="_blank" rel="noopener noreferrer">
                        <h3>{results[id]["title"]}</h3>
                        <p>{results[id]["summary"]}</p>
                    </a>
                </div>
            ))}
        </div>
    );
};  