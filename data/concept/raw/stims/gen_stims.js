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


// Rules from Cultural Ratchet Experiment
var genRules = function() {
	var single_feature_concepts = [
		{
			[constants.name]: 'flowers_orange_stems',
			[constants.phrase]: 'flowers with orange stems',
			[constants.type]: constants.single_feature,
			[constants.description]: {
				[constants.creature]: constants.flower,
				[constants.stem_color]: constants.orange,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return (
                    creature_description[constants.stem_color] === concept_description[constants.stem_color]
                )
            },
        },
		{
			[constants.name]: 'flowers_thorns',
			[constants.phrase]: 'flowers with thorns',
			[constants.type]: constants.single_feature,
			[constants.description]: {
				[constants.creature]: constants.flower,
				[constants.thorns_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return (
                    creature_description[constants.thorns_present] === concept_description[constants.thorns_present]
                )
            },
        },
		{
			[constants.name]: 'fish_fangs',
			[constants.phrase]: 'fish with white fangs',
			[constants.type]: constants.single_feature,
			[constants.description]: {
				[constants.creature]: constants.fish,
                [constants.fangs_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return (
                    creature_description[constants.fangs_present] === concept_description[constants.fangs_present]
                )
            },
        },
		{
			[constants.name]: 'fish_whiskers',
			[constants.phrase]: 'fish with whiskers',
			[constants.type]: constants.single_feature,
			[constants.description]: {
				[constants.creature]: constants.fish,
                [constants.whiskers_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return (
                    creature_description[constants.whiskers_present] === concept_description[constants.whiskers_present]
                )
            },
        },
		{
			[constants.name]: 'bugs_orange_head',
			[constants.phrase]: 'bugs with white orange head',
			[constants.type]: constants.single_feature,
			[constants.description]: {
				[constants.creature]: constants.bug,
                [constants.head_color]: constants.orange,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return (
                    creature_description[constants.head_color] === concept_description[constants.head_color]
                )
            },
        },
		{
			[constants.name]: 'bugs_without_wings',
			[constants.phrase]: 'bugs without wings',
			[constants.type]: constants.single_feature,
			[constants.description]: {
				[constants.creature]: constants.bug,
                [constants.wings_present]: constants.false,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return (
                    creature_description[constants.wings_present] === concept_description[constants.wings_present]
                )
            },
        },
		
		{
			[constants.name]: 'birds_tails',
			[constants.phrase]: 'birds with tails',
			[constants.type]: constants.single_feature,
			[constants.description]: {
				[constants.creature]: constants.bird,
                [constants.tail_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return (
                    creature_description[constants.tail_present] === concept_description[constants.tail_present]
                )
            },
        }, 
		{
			[constants.name]: 'birds_purple_tails',
			[constants.phrase]: 'birds with purple tails',
			[constants.type]: constants.single_feature,
			[constants.description]: {
				[constants.creature]: constants.bird,
                [constants.tail_present]: constants.true,
                [constants.crest_tail_color]: constants.purple
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return (
                    creature_description[constants.tail_present] === concept_description[constants.tail_present] &&
                    creature_description[constants.crest_tail_color] === concept_description[constants.crest_tail_color]
                )
            },
        }, 
	
		{
			[constants.name]: 'trees_purple_berries',
			[constants.phrase]: 'trees with purple berries',
			[constants.type]: constants.single_feature,
			[constants.description]: {
				[constants.creature]: constants.tree,
                [constants.berries_present]: constants.true,
                [constants.berries_color]: constants.purple,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return (
                    creature_description[constants.berries_present] === concept_description[constants.berries_present] &&
                    creature_description[constants.berries_color] === concept_description[constants.berries_color]
                )
            },
        },
		{
			[constants.name]: 'trees_without_leaves',
			[constants.phrase]: 'trees without leaves',
			[constants.type]: constants.single_feature,
			[constants.description]: {
				[constants.creature]: constants.tree,
                [constants.leaves_present]: constants.false,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return (
                    creature_description[constants.leaves_present] === concept_description[constants.leaves_present]
                )
            },
        },
	];

    var conjunction_concepts = [
        {
			[constants.name]: 'fish_orange_bodies_purple_stripes',
			[constants.phrase]: 'fish with orange bodies and purple stripes',
			[constants.type]: constants.conjunction,
			[constants.description]: {
				[constants.creature]: constants.fish,
                [constants.body_color]: constants.orange,
                [constants.stripes_present]: constants.true,
                [constants.stripes_color]: constants.purple,
            },
        },
        {
			[constants.name]: 'fish_white_stripes_whiskers',
			[constants.phrase]: 'fish with white stripes and whiskers',
			[constants.type]: constants.conjunction,
			[constants.description]: {
				[constants.creature]: constants.fish,
                [constants.whiskers_present]: constants.true,
                [constants.stripes_present]: constants.true,
                [constants.stripes_color]: constants.white,
            },
        },
    
        {
			[constants.name]: 'bugs_purple_legs_white_heads',
			[constants.phrase]: 'bugs with purple legs and white heads',
			[constants.type]: constants.conjunction,
			[constants.description]: {
				[constants.creature]: constants.bug,
                [constants.legs_color]: constants.purple,
                [constants.head_color]: constants.white,
            },
        },
        {
			[constants.name]: 'bugs_wings_antennae',
			[constants.phrase]: 'bugs with wings and antennae',
			[constants.type]: constants.conjunction,
			[constants.description]: {
				[constants.creature]: constants.bug,
                [constants.wings_present]: constants.true,
                [constants.antennae_present]: constants.true,
            },
        },
        {
			[constants.name]: 'birds_purple_wings_crests',
			[constants.phrase]: 'birds with purple wings and crests',
			[constants.type]: constants.conjunction,
			[constants.description]: {
				[constants.creature]: constants.bird,
                [constants.bird_wing_color]: constants.purple,
                [constants.crest_present]: constants.true,
            },
        },
        {
			[constants.name]: 'birds_orange_wings_tails',
			[constants.phrase]: 'birds with orange wings and tails',
			[constants.type]: constants.conjunction,
			[constants.description]: {
				[constants.creature]: constants.bird,
                [constants.bird_wing_color]: constants.orange,
                [constants.tail_present]: constants.true,
            },
        },
		{
			[constants.name]: 'trees_orange_trunks_berries',
			[constants.phrase]: 'trees with orange trunks and berries',
			[constants.type]: constants.conjunction,
			[constants.description]: {
				[constants.creature]: constants.tree,
                [constants.berries_present]: constants.true,
                [constants.trunk_color]: constants.orange,
            },
        },
		{
			[constants.name]: 'trees_purple_leaves_berries',
			[constants.phrase]: 'trees with purple leaves and berries',
			[constants.type]: constants.conjunction,
			[constants.description]: {
				[constants.creature]: constants.tree,
                [constants.berries_present]: constants.true,
                [constants.leaves_present]: constants.true,
                [constants.leaves_color]: constants.purple,
            },
        },
        {
			[constants.name]: 'flowers_purple_stems_thorns',
			[constants.phrase]: 'flowers with purple stem and thorns',
			[constants.type]: constants.conjunction,
			[constants.description]: {
				[constants.creature]: constants.flower,
                [constants.stem_color]: constants.purple,
                [constants.thorns_present]: constants.true,
            },
        },
        {
            [constants.name]: 'flowers_orange_petals_purple_centers',
            [constants.phrase]: 'flowers with orange petals and purple centers',
            [constants.type]: constants.conjunction,
            [constants.description]: {
                [constants.creature]: constants.flower,
                [constants.petals_color]: constants.orange,
                [constants.center_color]: constants.purple,
            },
        },      
    ];

    var disjunction_concepts = [
        {
            [constants.name]: 'bugs_orange_antennae_or_wings',
            [constants.phrase]: 'bugs with orange antennae or wings',
            [constants.type]: constants.disjunction,
            [constants.description]: {
                [constants.creature]: constants.bug,
                [constants.antennae_present]: constants.true,
                [constants.antennae_color]: constants.orange,
                [constants.wings_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (
                        creature_description[constants.antennae_present] == concept_description[constants.antennae_present] &&
                        creature_description[constants.antennae_color] == concept_description[constants.antennae_color]    
                    )||
                    creature_description[constants.wings_present] == concept_description[constants.wings_present]
                )
            }
        },
        {
            [constants.name]: 'bugs_purple_wings_or_white_legs',
            [constants.phrase]: 'bugs with purple wings or white legs',
            [constants.type]: constants.disjunction,
            [constants.description]: {
                [constants.creature]: constants.bug,
                [constants.legs_color]: constants.white,
                [constants.wings_present]: constants.true,
                [constants.bug_wings_color]: constants.purple,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    creature_description[constants.legs_color] == concept_description[constants.legs_color] ||
                    (creature_description[constants.wings_present] == concept_description[constants.wings_present] &&
                    creature_description[constants.bug_wings_color] == concept_description[constants.bug_wings_color])
                )
            }
        },
        {
            [constants.name]: 'birds_orange_tails_or_white_wings',
            [constants.phrase]: 'birds with orange tails or white wings',
            [constants.type]: constants.disjunction,
            [constants.description]: {
                [constants.creature]: constants.bird,
                [constants.bird_wing_color]: constants.white,
                [constants.tail_present]: constants.true,
                [constants.crest_tail_color]: constants.orange,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    creature_description[constants.bird_wing_color] == concept_description[constants.bird_wing_color] ||
                    (creature_description[constants.tail_present] == concept_description[constants.tail_present] &&
                    creature_description[constants.crest_tail_color] == concept_description[constants.crest_tail_color])
                )
            }
        },
        {
            [constants.name]: 'birds_orange_crests_or_purple_wings',
            [constants.phrase]: 'birds with orange crests or purple wings',
            [constants.type]: constants.disjunction,
            [constants.description]: {
                [constants.creature]: constants.bird,
                [constants.bird_wing_color]: constants.purple,
                [constants.crest_present]: constants.true,
                [constants.crest_tail_color]: constants.orange,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    creature_description[constants.bird_wing_color] == concept_description[constants.bird_wing_color] ||
                    (creature_description[constants.crest_present] == concept_description[constants.crest_present] &&
                    creature_description[constants.crest_tail_color] == concept_description[constants.crest_tail_color])
                )
            }
        },
        {
            [constants.name]: 'trees_purple_berries_or_white_trunks',
            [constants.phrase]: 'trees with purple berries or white trunks',
            [constants.type]: constants.disjunction,
            [constants.description]: {
                [constants.creature]: constants.tree,
                [constants.berries_present]: constants.true,
                [constants.berries_color]: constants.purple,
                [constants.trunk_color]: constants.white,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    creature_description[constants.trunk_color] == concept_description[constants.trunk_color] ||
                    (creature_description[constants.berries_present] == concept_description[constants.berries_present] &&
                    creature_description[constants.berries_color] == concept_description[constants.berries_color])
                )
            }
        },
        {
            [constants.name]: 'trees_white_leaves_or_orange_berries',
            [constants.phrase]: 'trees with white leaves or orange berries',
            [constants.type]: constants.disjunction,
            [constants.description]: {
                [constants.creature]: constants.tree,
                [constants.berries_present]: constants.true,
                [constants.leaves_present]: constants.true,
                [constants.leaves_color]: constants.white,
                [constants.berries_color]: constants.orange,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.leaves_present] == concept_description[constants.leaves_present] &&
                    creature_description[constants.leaves_color] == concept_description[constants.leaves_color]) ||
                    (creature_description[constants.berries_present] == concept_description[constants.berries_present] &&
                    creature_description[constants.berries_color] == concept_description[constants.berries_color]
                    )
                )
            }
        },
        {
            [constants.name]: 'flowers_purple_petals_or_thorns',
            [constants.phrase]: 'flowers with purple petals or thorns',
            [constants.type]: constants.disjunction,
            [constants.description]: {
                [constants.creature]: constants.flower,
                [constants.petals_color]: constants.purple,
                [constants.thorns_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    creature_description[constants.petals_color] == concept_description[constants.petals_color] ||
                    creature_description[constants.thorns_present] == concept_description[constants.thorns_present]
                )
            }
        },
        {
            [constants.name]: 'flowers_orange_stems_or_thorns',
            [constants.phrase]: 'flowers with orange stems or thorns',
            [constants.type]: constants.disjunction,
            [constants.description]: {
                [constants.creature]: constants.flower,
                [constants.stem_color]: constants.orange,
                [constants.thorns_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    creature_description[constants.stem_color] == concept_description[constants.stem_color] ||
                    creature_description[constants.thorns_present] == concept_description[constants.thorns_present]
                )
            }
        },
        {
            [constants.name]: 'fish_orange_bodies_or_fangs',
            [constants.phrase]: 'fish with orange bodies or fangs',
            [constants.type]: constants.disjunction,
            [constants.description]: {
                [constants.creature]: constants.fish,
                [constants.body_color]: constants.orange,
                [constants.fangs_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    creature_description[constants.body_color] == concept_description[constants.body_color] ||
                    creature_description[constants.fangs_present] == concept_description[constants.fangs_present]
                )
            }
        },
        {
            [constants.name]: 'fish_white_stripes_or_whiskers',
            [constants.phrase]: 'fish with white stripes or whiskers',
            [constants.type]: constants.disjunction,
            [constants.description]: {
                [constants.creature]: constants.fish,
                [constants.stripes_present]: constants.true,
                [constants.stripes_color]: constants.white,
                [constants.whiskers_present]: constants.true
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.stripes_color] == concept_description[constants.stripes_color] &&
                    creature_description[constants.stripes_present] == concept_description[constants.stripes_present]) ||
                    creature_description[constants.whiskers_present] == concept_description[constants.whiskers_present]
                )
            }
        },
    ];

    var conjunction_conjunction_concepts = [
        {
            [constants.name]: 'birds_purple_wings_white_crests_white_tails',
            [constants.phrase]: 'birds with purple wings and white crests and white tails',
            [constants.type]: constants.conjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.bird,
                [constants.bird_wing_color]: constants.purple,
                [constants.tail_present]: constants.true,
                [constants.crest_present]: constants.true,
                [constants.crest_tail_color]: constants.white,
            },
        },
        {
            [constants.name]: 'birds_crests_tails_orange_wings',
            [constants.phrase]: 'birds with crests and tails and orange wings',
            [constants.type]: constants.conjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.bird,
                [constants.bird_wing_color]: constants.orange,
                [constants.tail_present]: constants.true,
                [constants.crest_present]: constants.true,
            },
        },
        {
            [constants.name]: 'trees_orange_berries_purple_trunks_white_leaves',
            [constants.phrase]: 'Tree with orange berries and purple trunks and white leaves',
            [constants.type]: constants.conjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.tree,
                [constants.berries_present]: constants.true,
                [constants.leaves_present]: constants.true,
                [constants.berries_color]: constants.orange,
                [constants.trunk_color]: constants.purple,
                [constants.leaves_color]: constants.white,
            },
        },
        {
            [constants.name]: 'trees_leaves_berries_orange_trunks',
            [constants.phrase]: 'trees with leaves and berries and orange trunks',
            [constants.type]: constants.conjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.tree,
                [constants.berries_present]: constants.true,
                [constants.leaves_present]: constants.true,
                [constants.trunk_color]: constants.orange,
            },
        },
    ];

    var conjunction_disjunction_concepts = [
        {
            [constants.name]: 'birds_purple_wings_white_crests_or_white_tails',
            [constants.phrase]: 'birds with (purple wings and white crests) or white tails',
            [constants.type]: constants.conjunction_disjunction,
            [constants.description]: {
                [constants.creature]: constants.bird,
                [constants.bird_wing_color]: constants.purple,
                [constants.tail_present]: constants.true,
                [constants.crest_tail_color]: constants.white,
                [constants.crest_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.bird_wing_color] == concept_description[constants.bird_wing_color] &&
                    creature_description[constants.crest_tail_color] == concept_description[constants.crest_tail_color] &&
                    creature_description[constants.crest_present] == concept_description[constants.crest_present]) ||
                    (creature_description[constants.tail_present] == concept_description[constants.tail_present] &&
                    creature_description[constants.crest_tail_color] == concept_description[constants.crest_tail_color])
                )
            }
        },
        {
            [constants.name]: 'birds_purple_crests_purple_tails_or_orange_wings',
            [constants.phrase]: 'birds with (purple crests and purple tails) or orange wings',
            [constants.type]: constants.conjunction_disjunction,
            [constants.description]: {
                [constants.creature]: constants.bird,
                [constants.bird_wing_color]: constants.orange,
                [constants.tail_present]: constants.true,
                [constants.crest_present]: constants.true,
                [constants.crest_tail_color]: constants.purple,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return (
                    (
                        creature_description[constants.tail_present] == concept_description[constants.tail_present] &&
                        creature_description[constants.crest_present] == concept_description[constants.crest_present] &&
                        creature_description[constants.crest_tail_color] == concept_description[constants.crest_tail_color]
                    ) ||
                    creature_description[constants.bird_wing_color] == concept_description[constants.bird_wing_color]
                )
            }
        },
        {
            [constants.name]: 'trees_orange_berries_purple_trunks_or_white_leaves',
            [constants.phrase]: 'Tree with (orange berries and purple trunks) or white leaves',
            [constants.type]: constants.conjunction_disjunction,
            [constants.description]: {
                [constants.creature]: constants.tree,
                [constants.berries_present]: constants.true,
                [constants.berries_color]: constants.orange,
                [constants.trunk_color]: constants.purple,
                [constants.leaves_present]: constants.true,
                [constants.leaves_color]: constants.white,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.trunk_color] == concept_description[constants.trunk_color] &&
                    creature_description[constants.berries_present] == concept_description[constants.berries_present] &&
                    creature_description[constants.berries_color] == concept_description[constants.berries_color]) ||
                    (creature_description[constants.leaves_present] == concept_description[constants.leaves_present] &&
                    creature_description[constants.leaves_color] == concept_description[constants.leaves_color])
                )
            }
        },
        {
            [constants.name]: 'trees_leaves_berries_or_orange_trunks',
            [constants.phrase]: 'trees with (leaves and berries) or orange trunks',
            [constants.type]: constants.conjunction_disjunction,
            [constants.description]: {
                [constants.creature]: constants.tree,
                [constants.berries_present]: constants.true,
                [constants.trunk_color]: constants.orange,
                [constants.leaves_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    creature_description[constants.trunk_color] == concept_description[constants.trunk_color] ||
                    (creature_description[constants.leaves_present] == concept_description[constants.leaves_present] &&
                    creature_description[constants.berries_present] == concept_description[constants.berries_present])
                )
            }
        },
        {
            [constants.name]: 'flowers_purple_stems_white_petals_or_orange_centers',
            [constants.phrase]: 'flowers with (purple stems and white petals) or orange centers',
            [constants.type]: constants.conjunction_disjunction,
            [constants.description]: {
                [constants.creature]: constants.flower,
                [constants.stem_color]: constants.purple,
                [constants.petals_color]: constants.white,
                [constants.center_color]: constants.orange,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.stem_color] == concept_description[constants.stem_color] &&
                    creature_description[constants.petals_color] == concept_description[constants.petals_color])||
                    creature_description[constants.center_color] == concept_description[constants.center_color]
                )
            }
            
        },
        {
            [constants.name]: 'flowers_thorns_purple_petals_or_orange_stems',
            [constants.phrase]: 'flowers with (thorns and purple petals) or orange stems',
            [constants.type]: constants.conjunction_disjunction,
            [constants.description]: {
                [constants.creature]: constants.flower,
                [constants.stem_color]: constants.orange,
                [constants.petals_color]: constants.purple,
                [constants.thorns_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.thorns_present] == concept_description[constants.thorns_present] &&
                    creature_description[constants.petals_color] == concept_description[constants.petals_color])||
                    creature_description[constants.stem_color] == concept_description[constants.stem_color]
                )
            }
            
        },
        {
            [constants.name]: 'fish_orange_bodies_purple_stripes_or_whiskers',
            [constants.phrase]: 'fish with (orange bodies and purple stripes) or whiskers',
            [constants.type]: constants.conjunction_disjunction,
            [constants.description]: {
                [constants.creature]: constants.fish,
                [constants.stripes_present]: constants.true,
                [constants.stripes_color]: constants.purple,
                [constants.body_color]: constants.orange,
                [constants.whiskers_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.stripes_color] == concept_description[constants.stripes_color] &&
                    creature_description[constants.stripes_present] == concept_description[constants.stripes_present] &&
                    creature_description[constants.body_color] == concept_description[constants.body_color]) ||
                    creature_description[constants.whiskers_present] == concept_description[constants.whiskers_present]
                )
            }
        },
        {
            [constants.name]: 'fish_white_bodies_orange_stripes_or_fangs',
            [constants.phrase]: 'fish with (white bodies and orange stripes) or fangs',
            [constants.type]: constants.conjunction_disjunction,
            [constants.description]: {
                [constants.creature]: constants.fish,
                [constants.stripes_present]: constants.true,
                [constants.stripes_color]: constants.orange,
                [constants.body_color]: constants.white,
                [constants.fangs_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.stripes_color] == concept_description[constants.stripes_color] &&
                    creature_description[constants.stripes_present] == concept_description[constants.stripes_present] &&
                    creature_description[constants.body_color] == concept_description[constants.body_color]) ||
                    creature_description[constants.fangs_present] == concept_description[constants.fangs_present]
                )
            }
        },
        {
            [constants.name]: 'bugs_purple_legs_white_heads_or_orange_wings',
            [constants.phrase]: 'bugs with (purple legs and white heads) or orange wings',
            [constants.type]: constants.conjunction_disjunction,
            [constants.description]: {
                [constants.creature]: constants.bug,
                [constants.legs_color]: constants.purple,
                [constants.wings_present]: constants.true,
                [constants.bug_wings_color]: constants.orange,
                [constants.head_color]: constants.white,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.legs_color] == concept_description[constants.legs_color] && 
                    creature_description[constants.head_color] == concept_description[constants.head_color])||
                    (creature_description[constants.wings_present] == concept_description[constants.wings_present] &&
                    creature_description[constants.bug_wings_color] == concept_description[constants.bug_wings_color])
                )
            }
        },
        {
            [constants.name]: 'bugs_white_legs_purple_wings_or_orange_antennae',
            [constants.phrase]: 'bugs with (white legs and purple wings) or orange antennae',
            [constants.type]: constants.conjunction_disjunction,
            [constants.description]: {
                [constants.creature]: constants.bug,
                [constants.legs_color]: constants.white,
                [constants.wings_present]: constants.true,
                [constants.bug_wings_color]: constants.purple,
                [constants.antennae_present]: constants.true,
                [constants.antennae_color]: constants.orange,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.legs_color] == concept_description[constants.legs_color] && 
                    creature_description[constants.wings_present] == concept_description[constants.wings_present] &&
                    creature_description[constants.bug_wings_color] == concept_description[constants.bug_wings_color])||
                    (creature_description[constants.antennae_present] == concept_description[constants.antennae_present] &&
                    creature_description[constants.antennae_color] == concept_description[constants.antennae_color])
                )
            }
        },
    ];

    var disjunction_conjunction_concepts = [
        {
            [constants.name]: 'trees_purple_trunks_or_white_leaves_orange_berries',
            [constants.phrase]: 'trees with (purple trunks or white leaves) and orange berries',
            [constants.type]: constants.disjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.tree,
                [constants.berries_present]: constants.true,
                [constants.berries_color]: constants.orange,
                [constants.trunk_color]: constants.purple,
                [constants.leaves_present]: constants.true,
                [constants.leaves_color]: constants.white,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.trunk_color] == concept_description[constants.trunk_color] ||
                        (creature_description[constants.leaves_present] == concept_description[constants.leaves_present] &&
                        creature_description[constants.leaves_color] == concept_description[constants.leaves_color])) &&
                    (creature_description[constants.berries_color] == concept_description[constants.berries_color] &&
                    creature_description[constants.berries_present] == concept_description[constants.berries_present])
                )
            }
        },
        {
            [constants.name]: 'trees_orange_trunks_or_berries_white_leaves',
            [constants.phrase]: 'trees with (orange trunks or berries) and white leaves',
            [constants.type]: constants.disjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.tree,
                [constants.berries_present]: constants.true,
                [constants.trunk_color]: constants.orange,
                [constants.leaves_present]: constants.true,
                [constants.leaves_color]: constants.white,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.trunk_color] == concept_description[constants.trunk_color] ||
                    creature_description[constants.berries_present] == concept_description[constants.berries_present]) &&
                    (creature_description[constants.leaves_color] == concept_description[constants.leaves_color] &&
                    creature_description[constants.leaves_present] == concept_description[constants.leaves_present])
                )
            }
        },
        {
            [constants.name]: 'flowers_purple_centers_or_orange_stems_with_thorns',
            [constants.phrase]: 'flowers with (purple centers or orange stems) and thorns',
            [constants.type]: constants.disjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.flower,
                [constants.stem_color]: constants.orange,
                [constants.center_color]: constants.purple,
                [constants.thorns_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    creature_description[constants.thorns_present] == concept_description[constants.thorns_present] &&
                    (creature_description[constants.center_color] == concept_description[constants.center_color] ||
                    creature_description[constants.stem_color] == concept_description[constants.stem_color])
                )
            }
        },
        {
            [constants.name]: 'flowers_purple_stems_or_thorns_white_centers',
            [constants.phrase]: 'flowers with (purple stems or thorns) and white centers',
            [constants.type]: constants.disjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.flower,
                [constants.stem_color]: constants.purple,
                [constants.center_color]: constants.white,
                [constants.thorns_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    creature_description[constants.center_color] == concept_description[constants.center_color] &&
                    (creature_description[constants.stem_color] == concept_description[constants.stem_color] ||
                    creature_description[constants.thorns_present] == concept_description[constants.thorns_present])
                )
            }
        },
        {
            [constants.name]: 'fish_orange_bodies_or_fangs_whiskers',
            [constants.phrase]: 'fish with (orange bodies or fangs) and whiskers',
            [constants.type]: constants.disjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.fish,
                [constants.fangs_present]: constants.true,
                [constants.body_color]: constants.orange,
                [constants.whiskers_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.body_color] == concept_description[constants.body_color] ||
                    creature_description[constants.fangs_present] == concept_description[constants.fangs_present]) &&
                    creature_description[constants.whiskers_present] == concept_description[constants.whiskers_present]
                )
            }
        },
        {
            [constants.name]: 'fish_white_stripes_or_purple_bodies_whiskers',
            [constants.phrase]: 'fish with (white stripes or purple bodies) and whiskers',
            [constants.type]: constants.disjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.fish,
                [constants.stripes_present]: constants.true,
                [constants.stripes_color]: constants.white,
                [constants.body_color]: constants.purple,
                [constants.whiskers_present]: constants.true,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.body_color] == concept_description[constants.body_color] ||
                        (creature_description[constants.stripes_present] == concept_description[constants.stripes_present] &&
                        creature_description[constants.stripes_color] == concept_description[constants.stripes_color])) &&
                    creature_description[constants.whiskers_present] == concept_description[constants.whiskers_present]
                )
            }
        },
        {
            [constants.name]: 'bugs_antennae_or_wings_purple_bodies',
            [constants.phrase]: 'bugs with (antennae or wings) and purple bodies',
            [constants.type]: constants.disjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.bug,
                [constants.antennae_present]: constants.true,
                [constants.wings_present]: constants.true,
                [constants.body_color]: constants.purple,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.antennae_present] == concept_description[constants.antennae_present] || 
                    creature_description[constants.wings_present] == concept_description[constants.wings_present]) && 
                    creature_description[constants.body_color] == concept_description[constants.body_color]
                )
            }
        },
        {
            [constants.name]: 'bugs_white_heads_or_orange_antennae_purple_legs',
            [constants.phrase]: 'bugs with (white heads or orange antennae) and purple legs',
            [constants.type]: constants.disjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.bug,
                [constants.antennae_present]: constants.true,
                [constants.antennae_color]: constants.orange,
                [constants.head_color]: constants.white,
                [constants.legs_color]: constants.purple,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.head_color] == concept_description[constants.head_color] || 
                        (creature_description[constants.antennae_present] == concept_description[constants.antennae_present] &&
                        creature_description[constants.antennae_color] == concept_description[constants.antennae_color])) && 
                    creature_description[constants.legs_color] == concept_description[constants.legs_color]
                )
            }
        },
        {
            [constants.name]: 'birds_orange_tails_or_white_wings_orange_crests',
            [constants.phrase]: 'birds with (orange tails or white wings) and orange crests',
            [constants.type]: constants.disjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.bird,
                [constants.bird_wing_color]: constants.white,
                [constants.tail_present]: constants.true,
                [constants.crest_present]: constants.true,
                [constants.crest_tail_color]: constants.orange,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.bird_wing_color] == concept_description[constants.bird_wing_color] ||
                        (creature_description[constants.tail_present] == concept_description[constants.tail_present] &&
                        creature_description[constants.crest_tail_color] == concept_description[constants.crest_tail_color])) &&
                    (
                        creature_description[constants.crest_tail_color] == concept_description[constants.crest_tail_color] &&
                        creature_description[constants.crest_present] == concept_description[constants.crest_present]
                    )
                )
            }
        },
        {
            [constants.name]: 'birds_white_crests_or_orange_wings_tails',
            [constants.phrase]: 'birds with (white crests or orange wings) and tails',
            [constants.type]: constants.disjunction_conjunction,
            [constants.description]: {
                [constants.creature]: constants.bird,
                [constants.bird_wing_color]: constants.orange,
                [constants.tail_present]: constants.true,
                [constants.crest_present]: constants.true,
                [constants.crest_tail_color]: constants.white,
            },
            [constants.rule]: function(creature_type, creature_description, concept_description) {
                if (concept_description[constants.creature] !== creature_type) return false;
                return(
                    (creature_description[constants.bird_wing_color] == concept_description[constants.bird_wing_color] ||
                        (creature_description[constants.crest_present] == concept_description[constants.crest_present] &&
                        creature_description[constants.crest_tail_color] == concept_description[constants.crest_tail_color])) &&
                    creature_description[constants.tail_present] == concept_description[constants.tail_present]
                )
            }
        },
    ];

    return _.concat(
        single_feature_concepts,
        conjunction_concepts,
        disjunction_concepts,
        conjunction_disjunction_concepts,
        disjunction_conjunction_concepts
    );    
}

function genSalientFeatureMapping() {
    var rules = genRules();
    var salientFeaturesByRule = _.reduce(rules, function(result, rule){
        var description = rule[constants.description];
        var creature = description[constants.creature]
        var rule = {
            name: rule[constants.name],
            properties: [],
        };
        for (var creature_property in description) {
            if(creature_property == constants.creature) {
                continue;
            }
            var property_type = creature_dict[creature][creature_property].type;
            if (property_type === constants.bool) {
                var property_name = util.format('%s-%s', creature, creature_property); 
                rule.properties.push(property_name);
            } else if (property_type === constants.color) {
                var creature_color = description[creature_property];
                var property_name = util.format('%s-%s-%s', creature, creature_property, creature_color);
                rule.properties.push(property_name);
            }
        }
        result.push(rule)
        return result;
    }, []);

    var salientFeatures = _.uniq(_.reduce(salientFeaturesByRule, function(result, rule) {
        return _.concat(result, rule.properties);
    }, []));    

    fs.writeFile(
        './salient_feature_rule_mapping.json',
        JSON.stringify(salientFeaturesByRule),
        function(err)  {
            console.log(err);
        }
    );
    fs.writeFile(
        './salient_features.json',
        JSON.stringify(salientFeatures),
        function(err)  {
            console.log(err);
        }
    );
}

// -------------------------------------------------------
// Construct Vectorized Feature Representations of Stimuli
// -------------------------------------------------------

function constructFeatList() {
    var featList = [];
    for (var creature in creature_dict) {
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
            console.log(file);
            var fp = path.join(__dirname, input_dir, file);
            var stimuli = require(fp);
            var stimuliVecs = _.map(stimuli,  _.curry(constructFeatVector)(featList));
            var stimuliMap = Object.assign({}, stimuliVecs);
            var output_fp = path.join(__dirname, output_dir, file);
            jsonfile.writeFile(output_fp, stimuliMap);
        }
      });
}


// ----
// MAIN
// ----
// convertToVectorRepresentations('./train_stim/not_vectorized', './train_stim/vectorized');
genSalientFeatureMapping();
