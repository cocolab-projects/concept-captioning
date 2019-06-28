// -------------------------------------------------------
// Generate Critters for Multiplayer Concept Learning Game
// -------------------------------------------------------
// Here we generate flowers, fish, bugs, birds, and trees
// using Erin's Dominica-Bennett's stimuli generation package.
// -------------------------------------------------------

// -------
// IMPORTS
// -------
var _ = require('lodash');
var fs = require('fs');
var jsonfile = require('jsonfile');
var path = require('path');
var util = require('util');
var jsdom = require('jsdom');
const createCsvWriter = require('csv-writer').createObjectCsvWriter;  

// ----------
// CONSTANTS
// ----------
var constants = {
	// Flower
	flower: "flower",
	stem_color: "stem_color",
	spots_color: "spots_color",
	petals_color: "petals_color",
	center_color: "center_color",
	center_size: "center_size",
	petal_length: "petal_length",
	thorns_present: "thorns_present",
    spots_present: "spots_present",
    frill_present: "frill_present",
    frill_color: "frill_color",

	// Fish
	fish: "fish",
	body_color: "body_color",
	fins_color: "fins_color",
	body_size: "body_size",
	tail_size: "tail_size",
	fangs_present: "fangs_present",
    whiskers_present: "whiskers_present",
    stripes_present: "stripes_present",
    stripes_color: "stripes_color",

	// Bug
	bug: "bug",
	legs_color: "legs_color",
	head_color: "head_color",
	antennae_color: "antennae_color",
	bug_wings_color: "bug_wings_color",
	head_size: "head_size",
	antennae_present: "antennae_present",
	wings_present: "wings_present",

	// Bird
	bird: "bird",
	crest_tail_color: "crest_tail_color",
	bird_wing_color: "bird_wing_color",
	height: "height",
	fatness: "fatness",
	tail_present: "tail_present",
	crest_present: "crest_present",

	// Tree
	tree: "tree",
	berries_color: "berries_color",
	leaves_color: "leaves_color",
	trunk_color: "trunk_color",
	berries_present: "berries_present",
	leaves_present: "leaves_present",
    trunk_present: "trunk_present",
    trunk_width: "trunk_width",
    trunk_height: "trunk_height",

	// Property Names
	col1: "col1",
	col2: "col2",
	col3: "col3",
	col4: "col4",
    col5: "col5",
    col6: "col6",
	prop1: "prop1",
    prop2: "prop2",
	tar1: "tar1",
    tar2: "tar2",
    tar3: "tar3",

	// Property Types
	color: "color",
	size: "size",
	bool: "bool",

	// Colors
	blue: "blue",
	red: "red",
	yellow: "yellow",
	green: "green",
	orange: "orange",
	purple: "purple",
	pink: "pink",
	white: "white",
	black: "black",
	brown: "brown",
	
	// Sizes
	small: "small",
    large: "large",
    standard: "standard",

	// Boolean
	true: "true",
	false: "false",

    // Misc
    rule: "rule",
	creature: "creature",
	name: "name",
	type: "type",
	logical_form: "logical_form",
	description: "description",
	phrase: "phrase",
	props: "props",
	belongs_to_concept: "belongs_to_concept",

    // Rule Types
    single_feature: "SINGLE_FEATURE",
    conjunction: "CONJUNCTION",
    disjunction: "DISJUNCTION",
    conjunction_conjunction: "CONJUNCTION_CONJUNCTION",
    disjunction_disjunction: "DISJUNCTION_DISJUNCTION",
    conjunction_disjunction: "CONJUNCTION_DISJUNCTION",
    disjunction_conjunction: "DISJUNCTION_CONJUNCTION",
}


// --------------------------------------
// Creature Properties (Global Variables)
// --------------------------------------
var creature_dict = {
	[constants.flower]: {
		[constants.stem_color]: {
			[constants.name]: constants.col1,
			[constants.type]: constants.color,
		},
		[constants.spots_color]: {
			[constants.name]: constants.col2,
			[constants.type]: constants.color,
		},
		[constants.petals_color]: {
			[constants.name]: constants.col3,
			[constants.type]: constants.color,
		},
		[constants.center_color]: {
			[constants.name]: constants.col4,
			[constants.type]: constants.color,
        },
		[constants.frill_color]: {
			[constants.name]: constants.col5,
			[constants.type]: constants.color,
		},
		[constants.center_size]: {
			[constants.name]: constants.prop1,
			[constants.type]: constants.size,
		},
		[constants.petal_length]: {
			[constants.name]: constants.prop2,
			[constants.type]: constants.size,
		},
		[constants.thorns_present]: {
			[constants.name]: constants.tar1,
			[constants.type]: constants.bool,			
		},
		[constants.spots_present]: {
			[constants.name]: constants.tar2,
			[constants.type]: constants.bool,			
        },
		[constants.frill_present]: {
			[constants.name]: constants.tar3,
			[constants.type]: constants.bool,			
		},
	},

	[constants.fish]: {
		[constants.body_color]: {
			[constants.name]: constants.col1,
			[constants.type]: constants.color,
		},
		[constants.fins_color]: {
			[constants.name]: constants.col2,
			[constants.type]: constants.color,
		},
		[constants.body_size]: {
			[constants.name]: constants.prop1,
			[constants.type]: constants.size,
		},
		[constants.tail_size]: {
			[constants.name]: constants.prop2,
			[constants.type]: constants.size,
		},
		[constants.fangs_present]: {
			[constants.name]: constants.tar1,
			[constants.type]: constants.bool,			
		},
		[constants.whiskers_present]: {
			[constants.name]: constants.tar2,
			[constants.type]: constants.bool,			
        },
		[constants.stripes_present]: {
			[constants.name]: constants.tar3,
			[constants.type]: constants.bool,			
        },
		[constants.stripes_color]: {
			[constants.name]: constants.col3,
			[constants.type]: constants.color,			
		},
	},

	[constants.bug]: {
		[constants.legs_color]: {
			[constants.name]: constants.col1,
			[constants.type]: constants.color,
		},
		[constants.head_color]: {
			[constants.name]: constants.col2,
			[constants.type]: constants.color,
		},
		[constants.body_color]: {
			[constants.name]: constants.col3,
			[constants.type]: constants.color,
		},
		[constants.antennae_color]: {
			[constants.name]: constants.col4,
			[constants.type]: constants.color,
		},
		[constants.bug_wings_color]: {
			[constants.name]: constants.col5,
			[constants.type]: constants.color,
		},
		[constants.head_size]: {
			[constants.name]: constants.prop1,
			[constants.type]: constants.size,
		},
		[constants.body_size]: {
			[constants.name]: constants.prop2,
			[constants.type]: constants.size,
		},
		[constants.antennae_present]: {
			[constants.name]: constants.tar1,
			[constants.type]: constants.bool,			
		},
		[constants.wings_present]: {
			[constants.name]: constants.tar2,
			[constants.type]: constants.bool,			
		},
	},

	[constants.bird]: {
		[constants.crest_tail_color]: {
			[constants.name]: constants.col1,
			[constants.type]: constants.color,
		},
		[constants.body_color]: {
			[constants.name]: constants.col2,
			[constants.type]: constants.color,
		},
		[constants.bird_wing_color]: {
			[constants.name]: constants.col3,
			[constants.type]: constants.color,
		},
		[constants.height]: {
			[constants.name]: constants.prop1,
			[constants.type]: constants.size,
		},
		[constants.fatness]: {
			[constants.name]: constants.prop2,
			[constants.type]: constants.size,
		},
		[constants.tail_present]: {
			[constants.name]: constants.tar1,
			[constants.type]: constants.bool,			
		},
		[constants.crest_present]: {
			[constants.name]: constants.tar2,
			[constants.type]: constants.bool,			
		},
	},

	[constants.tree]: {
		[constants.berries_color]: {
			[constants.name]: constants.col1,
			[constants.type]: constants.color,
		},
		[constants.leaves_color]: {
			[constants.name]: constants.col2,
			[constants.type]: constants.color,
		},
		[constants.trunk_color]: {
			[constants.name]: constants.col3,
			[constants.type]: constants.color,
		},
		[constants.berries_present]: {
			[constants.name]: constants.tar1,
			[constants.type]: constants.bool,			
		},
		[constants.leaves_present]: {
			[constants.name]: constants.tar2,
			[constants.type]: constants.bool,			
        },
        [constants.trunk_width]: {
            [constants.name]: constants.prop1,
            [constants.type]: constants.size,
        },
        [constants.trunk_height]: {
            [constants.name]: constants.prop2,
            [constants.type]: constants.size,
        }
	},
};

var color_dict = {
	[constants.blue]: "#5da5db",
	[constants.red]: "#f42935",
	[constants.yellow]: "#eec900",
	[constants.green]: "#228b22",
	[constants.orange]: "#ff8c00",
	[constants.purple]: "#dda0dd",
	[constants.pink]: "#FF69B4",
	[constants.white]: "#FFFFFF",
	[constants.black]: "#000000",
	[constants.brown]: "#A52A2A",
};

var creature_to_colors_dict = {
    [constants.flower]: [constants.orange, constants.purple, constants.white],
    [constants.bug]: [constants.orange, constants.purple, constants.white],
    [constants.bird]: [constants.orange, constants.purple, constants.white],
    [constants.fish]: [constants.orange, constants.purple, constants.white],
    [constants.tree]: [constants.orange, constants.purple, constants.white, constants.green, constants.blue, constants.red],
};

// -----------------------------
// Construct Distractor Stimuli
// -------------------------------
function constructDistractors(target, salientFeatures) {
    var creature = target.creature;
    var possibleFeatures = _.intersection(
        salientFeatures,
        constructCreatureFeatureList(creature)
    );
    
    var distractorFeats =_.map(
        _.shuffle(possibleFeatures).slice(0, 2),
        generateFeature
    );

    var distractors = _.map(
        distractorFeats,
        _.curry(constructDistractor)(target)
    );

    return {
        'distractors': distractors,
        'distractorFeats': {
            'distr1': distractorFeats[0].join('-'),
            'distr2': distractorFeats[1].join('-'),
        },
    }
}

function generateFeature(feat) {
    var properties = _.split(feat, '-');
    if (properties.length == 2) {
        return [
            properties[0],
            properties[1],
             _.shuffle([constants.true, constants.false])[0]
        ];
    } else if (properties.length == 3) {
        return properties;
    } else {
        throw new Error("Improper feature construction");
    }
}

function constructDistractor(target, distractorFeat) {
    var d = _.cloneDeep(target);
    var creature = distractorFeat[0];
    d["description"][distractorFeat[1]] = distractorFeat[2];

    var propertyName = creature_dict[distractorFeat[0]][distractorFeat[1]]['name'];
    var propertyType = creature_dict[distractorFeat[0]][distractorFeat[1]]['type'];
    
    if (propertyType === constants.bool) {
        if (distractorFeat[2] === constants.true) {

            d["props"][propertyName] = true;

            // If necessary, set color property so that
            // both the speaker and listener have a critter
            // with the same color, for the property
            // that we are adding.
            var colorFeature = null;
            if (distractorFeat[1] == "stripes_present") {
                colorFeature = "stripes_color";
            } else if (distractorFeat[1] == "wings_present") {
                colorFeature = "bug_wings_color";                
            } else if (distractorFeat[1] == "antennae_present") {
                colorFeature = "antennae_color";
            } else if (distractorFeat[1] == "crest_present") {
                colorFeature = "crest_tail_color";           
            } else if (distractorFeat[1] == "leaves_present") {
                colorFeature = "leaves_color";     
            } else if (distractorFeat[1] == "tail_present") {
                colorFeature = "crest_tail_color";                
            } else if (distractorFeat[1] == "berries_present") {
                colorFeature = "berries_color";
            }
            if (colorFeature != null) {
                var color = _.shuffle(creature_to_colors_dict[creature])[0];
                d["description"][colorFeature] = color;
                var colorPropertyName = creature_dict[distractorFeat[0]][colorFeature]['name'];
                d["props"][colorPropertyName] = color_dict[color];
            }
        } else if (distractorFeat[2] === constants.false) {
            d["props"][propertyName] = false;
        } else {
            throw new Error ("Improper boolean value for distractor feature");
        }
    } else if (propertyType == constants.color) {
        // If necessary, set boolean property to true accordingly
        // Here, we only hard code these for the salient features
        // from the cultural ratchet dataset
        var boolProperties = [];
        if (distractorFeat[1] == "stripes_color") {
            boolProperties.push("stripes_present");
        } else if (distractorFeat[1] == "bug_wings_color") {
            boolProperties.push("wings_present");
        } else if (distractorFeat[1] == "antennae_color") {
            boolProperties.push("antennae_present");   
        } else if (distractorFeat[1] == "leaves_color") {
            boolProperties.push("leaves_present");   
        } else if (distractorFeat[1] == "crest_tail_color") {
            boolProperties = _.shuffle(
                [["crest_present"], ["tail_present"], ["crest_present", "tail_present"]]
            )[0];
        } else if (distractorFeat[1] == "berries_color") {
            boolProperties.push("berries_present");
        }
        for (var i = 0; i < boolProperties.length; i++) {
            var p = boolProperties[i];
            d["description"][p] = constants.true;
            var boolPropertyName = creature_dict[distractorFeat[0]][p]['name'];
            d["props"][boolPropertyName] = true;
        }

        d["props"][propertyName] = color_dict[distractorFeat[2]];
    } else {
        throw new Error("Invalid property type for distractor");
    }

    return d;
}

// -------------------------------------------------------
// Construct Vectorized Feature Representations of Stimuli
// -------------------------------------------------------
function constructFeatList() {
    var featList = [];
    for (var creature in creature_dict) {
        featList = _.concat(featList, constructCreatureFeatureList(creature));
    }
    return featList;
}

function constructCreatureFeatureList(creature){
    var featList = [];
    for (var creature_property in creature_dict[creature]) {
        var property_type = creature_dict[creature][creature_property].type;
        if (property_type === constants.bool) {
            var property_name = util.format('%s-%s', creature, creature_property); 
            featList.push(property_name);
        } else if (property_type === constants.color) {
            var creature_colors = creature_to_colors_dict[creature];
            for (var i = 0; i < creature_colors.length; i++) {
                var property_name = util.format('%s-%s-%s', creature, creature_property, creature_colors[i]);
                featList.push(property_name);
            }
        }
    }
    return featList;
}

function constructFeatVector(featList, stim){
    // Construct vector representation of a stimulus.
    // Here we store this as an object/dictionary,
    // where the same keys (items in featList) 
    // are shared across all stimuli.
    // Construct a list of binary features that may encapsulate all
    // the feature of stimuli in the dataset.
    var stimVec = featList.reduce((o, feat) => ({ ...o, [feat]: 0}), {}); // 0 default val
    var creature = stim['creature'];
    var description = stim['description'];
    for (var p in description) {
        if (description.hasOwnProperty(p)) {
            var v = description[p];
            var possibleProperty = '';
            if (v === constants.true || v === constants.false) {
                // boolean
                possibleProperty = util.format('%s-%s', creature, p);
            } else {
                // color
                possibleProperty = util.format('%s-%s-%s', creature, p, v);
            }
            if (possibleProperty in stimVec) {
                stimVec[possibleProperty] = 1;
            }
        }
    }
    return stimVec;
}

function convertToVectorRepresentations(input_dir, output_dir) {
	// Create dataset directories
	if (!fs.existsSync(path.join(__dirname, output_dir))){
		fs.mkdirSync(path.join(__dirname, output_dir));
	}

    // Feature list
    var featList = constructFeatList();

    // Construct vectorized stims
    fs.readdirSync(input_dir).forEach(file => {
        if (path.extname(file) == '.json') {
            var fp = path.join(__dirname, input_dir, file);
            var rounds = require(fp);
            var stims = [];
            for (var roundNum = 0; roundNum < rounds.length; roundNum++) {
                var roundStims = rounds[roundNum].stimuli;
                var roundStimVecs = _.map(roundStims, _.curry(constructFeatVector)(featList));
                var roundStimMap = {
                    "target": roundStimVecs[0],
                    "distr1": roundStimVecs[1],
                    "distr2": roundStimVecs[2],
                }
                stims.push(roundStimMap);
            }
            var stimuliMap = Object.assign({}, stims);
            var output_fp = path.join(__dirname, output_dir, file);
            jsonfile.writeFile(output_fp, stimuliMap);
        }
      });
}

// ---------------------------------
// Construct .png files for stimuli
// ---------------------------------

function stimToSVG(stim, filename) {
    // Given stimulus description (see concept data for example)
    // construct a SVG image and write this to disk.
    // Adapted from https://odino.org/generating-svgs-with-raphaeljs-and-nodejs/
    jsdom.env(
        "<html></html>",
        [],
        function (errors, win) {
          if (errors) {
            throw errors;
          } else {
            // create / override the global window
            // object, as raphael will access it
            // globally
            global.window = win
            global.document = win.document
            global.navigator = win.navigator
    
            var raphael = require('./js/raphael.js');
            raphael.setWindow(win)

            var genEcosystem = require('./js/ecosystem.js').genEcosystem;
            var Ecosystem = genEcosystem(raphael);
    
            var paper = raphael(0, 0, 250, 250);
            Ecosystem.draw(stim.creature, stim.props, paper);

            var svg = win.document.documentElement.innerHTML;
            svg = svg.replace('<head></head><body>', '');
            svg = svg.replace('</body>', '');

            fs.writeFile(filename, svg, (err) => {
                if (err) console.log(err);
            });
          }
        }
    );
}

function guidGenerator() {
    var S4 = function() {
       return (((1+Math.random())*0x10000)|0).toString(16).substring(1);
    };
    return (S4()+S4()+"-"+S4()+"-"+S4()+"-"+S4()+"-"+S4()+S4()+S4());
}

function genSVGsForStims(stimDescriptions, imgsDir) {
    var ids = []
    for (var i = 0; i < stimDescriptions.length; i++) {
        var id = guidGenerator() + '.svg';
        ids.push({'id': id});
        var imgFp = path.join(imgsDir, id);
        stimToSVG(stimDescriptions[i], imgFp);
    }
    return ids;
}


function genConceptSVG(input_dir, output_dir) {
    var stimsDir = path.join(input_dir, 'not_vectorized');
	if (!fs.existsSync(path.join(__dirname, output_dir))){
        fs.mkdirSync(path.join(__dirname, output_dir));
    }
    var imgsDir= path.join(__dirname, output_dir, 'svg_imgs');
	if (!fs.existsSync(imgsDir)){
        fs.mkdirSync(imgsDir);
	}
    var idsDir= path.join(__dirname, output_dir, 'ids');
	if (!fs.existsSync(idsDir)){
        fs.mkdirSync(idsDir);
	}

    fs.readdirSync(stimsDir).forEach(file => {
        if (path.extname(file) == '.json') {
            console.log("Creating svgs for " + file);
            var fp = path.join(stimsDir, file);
            var stimDescriptions = require(fp);
            var imgIds = genSVGsForStims(stimDescriptions, imgsDir);
            var imgIdsFp = path.join(idsDir, file);

            const csvWriter = createCsvWriter({  
                path: imgIdsFp,
                header: [
                  {id: 'id', title: 'id'},
                ]
              });
            csvWriter.writeRecords(imgIds).then(() => {});
        }
      });
}


function genRefSVG(input_dir, output_dir) {
	// Create dataset directories
	if (!fs.existsSync(path.join(__dirname, output_dir))){
		fs.mkdirSync(path.join(__dirname, output_dir));
    }
    var imgsDir= path.join(__dirname, output_dir, 'svg_imgs');
	if (!fs.existsSync(imgsDir)){
        fs.mkdirSync(imgsDir);
	}
    var idsDir= path.join(__dirname, output_dir, 'ids');
	if (!fs.existsSync(idsDir)){
        fs.mkdirSync(idsDir);
	}

    // Construct vectorized stims
    fs.readdirSync(input_dir).forEach(file => {
        if (path.extname(file) == '.json') {
            console.log("Creating svgs for " + file);
            var fp = path.join(__dirname, input_dir, file);
            var rounds = require(fp);
            var stims = [];
            for (var roundNum = 0; roundNum < rounds.length; roundNum++) {
                var roundStims = rounds[roundNum].stimuli;
                var ids = genSVGsForStims(roundStims, imgsDir);
                var roundStimMap = {
                    "target": ids[0].id,
                    "distr1": ids[1].id,
                    "distr2": ids[2].id,
                }
                stims.push(roundStimMap);
            }
            var stimuliMap = Object.assign({}, stims);
            var output_fp = path.join( idsDir, file);
            jsonfile.writeFile(output_fp, stimuliMap);
        }
      });
}


// --------
// MAIN
// --------
// convertToVectorRepresentations('../../data/reference/pilot_coll1/raw/trialList/', '../../data/reference/pilot_coll1/raw/stimuli')
// genConceptSVG('../../data/concept/raw/stims/test_stim', '../../data/concept/raw/stims/test_stim/vision');
genRefSVG('../../data/reference/pilot_coll1/raw/trialList/', '../../data/reference/pilot_coll1/raw/vision');

module.exports = {
    constructDistractors,
    convertToVectorRepresentations
};
