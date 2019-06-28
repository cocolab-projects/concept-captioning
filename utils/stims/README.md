# Stims
This directory contains utilities for generating representations of stimuli from both the concept and reference games.

## Stimulus Generation
First install node dependencies with `npm install`. Then select the appropriate method in `format_stimulus.js` from the following:

- JSON Params -> Binary feature vectors
    Example: `convertToVectorRepresentations('../../data/reference/pilot_coll1/raw/trialList/', '../../data/reference/pilot_coll1/raw/stimuli')`

- JSON Params -> SVG images
    Example: `genSVGsForDir('../../data/concept/raw/stims/test_stim', '../../data/concept/raw/stims/test_stim/vision');`

    To convert all svgs in a directory to pngs use https://www.npmjs.com/package/convert-svg-to-png.
    Example: `convert-svg-to-png * --background=white --width=250 --height=250`