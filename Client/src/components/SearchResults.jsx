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
                        <div className='title_'><h3>{results[id]["title"]}</h3></div>
                        <div className='desc'>
                            <p>{results[id]["summary"]}</p>
                        </div>
                        <div className='bar'></div>
                    </a>
                </div>
            ))}
        </div>
    );
};  