import os
import re
import json
from dmcontent.content_loader import ContentLoader


MANIFESTS = {
    'services': {
        'question_set': 'services',
        'manifest': 'edit_submission'
    },
    'briefs': {
        'question_set': 'briefs',
        'manifest': 'edit_brief'
    },
    'brief-responses': {
        'question_set': 'brief-responses',
        'manifest': 'edit_brief_response'
    }
}


SCHEMAS = {
    'services': [
        ('Digital Professionals Service',
         'digital-service-professionals', 'digital-professionals'),

        ('Digital Outcomes Service',
         'digital-service-professionals', 'digital-outcome'),

        ('Digital Professionals Service',
         'digital-marketplace', 'digital-professionals'),

        ('Digital Outcomes Service',
         'digital-marketplace', 'digital-outcome'),
    ],
    'briefs': [
        ('Digital Service Professionals Brief',
         'digital-service-professionals', 'digital-professionals'),

        ('Digital Outcome Brief',
         'digital-service-professionals', 'digital-outcome'),

        ('Digital Service Professionals Brief',
         'digital-marketplace', 'digital-professionals'),

        ('Digital Outcome Brief',
         'digital-marketplace', 'digital-outcome'),

        ('Training',
         'digital-marketplace', 'training'),
    ],
    'brief-responses': [
        ('Digital Service Professionals Brief Response',
         'digital-service-professionals', 'digital-professionals'),

        ('Digital Outcome Brief Response',
         'digital-service-professionals', 'digital-outcome'),

        ('Digital Service Professionals Brief Response',
         'digital-marketplace', 'digital-professionals'),

        ('Digital Outcome Brief Response',
         'digital-marketplace', 'digital-outcome'),
    ]
}
EMAIL_PATTERN = r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)'


def load_manifest(schema_type, framework_slug, lot_slug):
    loader = ContentLoader('./')
    loader.load_manifest(
        framework_slug,
        MANIFESTS[schema_type]['question_set'],
        MANIFESTS[schema_type]['manifest']
    )

    return loader.get_builder(framework_slug, MANIFESTS[schema_type]['manifest']).filter({'lot': lot_slug})


def load_questions(manifest):
    return {q['id']: q for q in
            sum((s.questions for s in manifest.sections), [])}


def drop_non_schema_questions(questions):
    # These questions are used to generate questions in the UI but are
    # not used to validate the response, so we remove them....
    for key in ['id', 'lot', 'lotName']:
        questions.pop(key, None)


def empty_schema(schema_name):
    return {
        "title": "{} Schema".format(schema_name),
        "$schema": "http://json-schema.org/schema#",
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "required": [],
    }


def text_property(question):
    data = {
        "type": "string",
        "minLength": 0 if question.get('optional') else 1,
    }

    format_limit = question.get('limits', {}).get('format')
    if format_limit:
        data['format'] = format_limit
        if (format_limit == 'email'):
            data['pattern'] = EMAIL_PATTERN

    data.update(parse_question_limits(question))

    return {question['id']: data}


def upload_property(question):
    data = {
        "items": {
            "type": "string",
            "minLength": 0 if question.get('optional') else 1
        },
        "type": "array"
    }

    format_limit = question.get('limits', {}).get('format')
    if format_limit:
        data['format'] = format_limit

    data.update(parse_question_limits(question))

    return {question['id']: data}


def uri_property(question):
    return {question['id']: {
        "type": "string",
        "format": "uri",
    }}


def checkbox_property(question):
    """
    Convert a checkbox question into JSON Schema.
    """
    return {question['id']: {
        "type": "array",
        "uniqueItems": True,
        "minItems": 0 if question.get('optional') else 1,
        "maxItems": len(question['options']),
        "items": {
            "enum": [
                option.get('value', option['label'])
                for option in question['options']
            ]
        }
    }}


def radios_property(question):
    return {question['id']: {
        "enum": [
            option.get('value', option['label'])
            for option in question['options']
        ]
    }}


def boolean_property(question):
    return {question['id']: {
        "type": "boolean"
    }}


def list_property(question):
    items = {
        "type": "string",
        "maxLength": 100,
        "pattern": "^(?:\\S+\\s+){0,9}\\S+$"
    }

    format_limit = question.get('limits', {}).get('format')
    if format_limit:
        items['format'] = format_limit
        if (format_limit == 'email'):
            items['pattern'] = EMAIL_PATTERN

    items.update(parse_question_limits(question, for_items=True))

    return {question['id']: {
        "type": "array",
        "minItems": 0 if question.get('optional') else 1,
        "maxItems": question.get('number_of_items', 10),
        "items": items
    }}


def boolean_list_property(question):
    return {question['id']: {
        "type": "array",
        "minItems": 0 if question.get('optional') else 1,
        "maxItems": question.get('number_of_items', 10),
        "items": {
            "type": "boolean"
        }
    }}


def price_string(optional):
    pattern = r"^\d{1,15}(?:\.\d{1,5})?$"
    if optional:
        pattern = r"^$|" + pattern
    return {
        "type": "string",
        "pattern": pattern,
    }


def pricing_property(question):
    pricing = {}
    if 'price' in question.fields:
        pricing[question.fields['price']] = price_string(
            'price' in question.get('optional_fields', [])
        )
    if 'minimum_price' in question.fields:
        pricing[question.fields['minimum_price']] = price_string(
            'minimum_price' in question.get('optional_fields', [])
        )
    if 'maximum_price' in question.fields:
        pricing[question.fields['maximum_price']] = price_string(
            'maximum_price' in question.get('optional_fields', [])
        )
    if 'price_unit' in question.fields:
        pricing[question.fields['price_unit']] = {
            "enum": [
                "Unit",
                "Person",
                "Licence",
                "User",
                "Device",
                "Instance",
                "Server",
                "Virtual machine",
                "Transaction",
                "Megabyte",
                "Gigabyte",
                "Terabyte"
            ]
        }
        if 'price_unit' in question.get('optional_fields', []):
            pricing[question.fields['price_unit']]['enum'].insert(0, "")
    if 'price_interval' in question.fields:
        pricing[question.fields['price_interval']] = {
            "enum": [
                "Second",
                "Minute",
                "Hour",
                "Day",
                "Week",
                "Month",
                "Quarter",
                "6 months",
                "Year"
            ]
        }
        if 'price_interval' in question.get('optional_fields', []):
            pricing[question.fields['price_interval']]['enum'].insert(0, "")

    if 'hours_for_price' in question.fields:
        pricing[question.fields['hours_for_price']] = {
            "enum": [
                "1 hour",
                "2 hours",
                "3 hours",
                "4 hours",
                "5 hours",
                "6 hours",
                "7 hours",
                "8 hours"
            ]
        }

    return pricing


def number_property(question):
    limits = question.get('limits', {})
    return {question['id']: {
        "exclusiveMaximum": not limits.get('integer_only'),
        "maximum": limits['max_value'] if limits.get('max_value') is not None else 100,
        "minimum": limits.get('min_value') or 0,
        "type": "integer" if limits.get('integer_only') else "number"
    }}


def multiquestion(question):
    """
    Moves subquestions of multiquestions into fully fledged questions.
    """
    properties = {}
    for nested_question in question['questions']:
        properties.update(build_question_properties(nested_question))

    return properties


QUESTION_TYPES = {
    'text': text_property,
    'upload': upload_property,  # uri_property requires http prefix etc.
    'textbox_large': text_property,
    'checkboxes': checkbox_property,
    'radios': radios_property,
    'boolean': boolean_property,
    'list': list_property,
    'boolean_list': boolean_list_property,
    'pricing': pricing_property,
    'pricing_aud': pricing_property,
    'pricing_gbp': pricing_property,
    'number': number_property,
    'multiquestion': multiquestion
}


def parse_question_limits(question, for_items=False):
    """
    Converts word and character length validators into JSON Schema-compatible maxLength and regex validators.
    """
    limits = {}
    word_length_validator = next(
        iter(filter(None, (
            re.match(r'under_(\d+)_words', validator['name'])
            for validator in question.get('validations', [])
        ))),
        None
    )
    char_length_validator = next(
        iter(filter(None, (
            re.search(r'(\d+)', validator['message'])
            for validator in question.get('validations', [])
            if validator['name'] == 'under_character_limit'
        ))),
        None
    )

    char_length = question.get('max_length') or (char_length_validator and char_length_validator.group(1))
    word_length = question.get('max_length_in_words') or (word_length_validator and word_length_validator.group(1))

    if char_length:
        limits['maxLength'] = int(char_length)

    if word_length:
        if not for_items and question.get('optional'):
            limits['pattern'] = r"^$|(^(?:\S+\s+){0,%s}\S+$)" % (int(word_length) - 1)
        else:
            limits['pattern'] = r"^(?:\S+\s+){0,%s}\S+$" % (int(word_length) - 1)

    return limits


def add_assurance(value_schema, assurance_approach):
    assurance_options = {
        '2answers-type1': [
            'Service provider assertion', 'Independent validation of assertion'
        ],
        '3answers-type1': [
            'Service provider assertion', 'Contractual commitment', 'Independent validation of assertion'
        ],
        '3answers-type2': [
            'Service provider assertion', 'Independent validation of assertion',
            'Independent testing of implementation'
        ],
        '3answers-type3': [
            'Service provider assertion', 'Independent testing of implementation', 'CESG-assured components'
        ],
        '3answers-type4': [
            'Service provider assertion', 'Independent validation of assertion',
            'Independent testing of implementation'
        ],
        '4answers-type1': [
            'Service provider assertion', 'Independent validation of assertion',
            'Independent testing of implementation', 'CESG-assured components'
        ],
        '4answers-type2': [
            'Service provider assertion', 'Contractual commitment',
            'Independent validation of assertion', 'CESG-assured components'
        ],
        '4answers-type3': [
            'Service provider assertion', 'Independent testing of implementation',
            'Assurance of service design', 'CESG-assured components'
        ],
        '5answers-type1': [
            'Service provider assertion', 'Contractual commitment', 'Independent validation of assertion',
            'Independent testing of implementation', 'CESG-assured components'
        ]
    }

    return {
        "type": "object",
        "properties": {
            "assurance": {
                "enum": assurance_options[assurance_approach]
            },
            "value": value_schema,
        },
        "required": [
            "value",
            "assurance"
        ]
    }


def build_question_properties(question):
    question_data = QUESTION_TYPES[question['type']](question)
    if question.get('assuranceApproach'):
        for key, value_schema in question_data.items():
            question_data[key] = add_assurance(value_schema, question['assuranceApproach'])
    return question_data


def build_any_of(any_of, fields):
    return {
        'required': [field for field in sorted(fields)],
        'title': any_of
    }


def build_schema_properties(schema, questions):
    for key, question in questions.items():
        schema['properties'].update(build_question_properties(question))
        schema['required'].extend(question.required_form_fields)

    schema['required'].sort()

    return schema


def add_multiquestion_anyof(schema, questions):
    any_ofs = {}

    for key, question in questions.items():
        if question.get('any_of'):
            question_fields = []
            for q in question.questions:
                if q.get('fields'):
                    question_fields.extend(val for val in q.get('fields').values())
                else:
                    question_fields.append(q.id)
            any_ofs[question.id] = build_any_of(question.get('any_of'), question_fields)

    if any_ofs:
        schema['anyOf'] = [any_ofs[key] for key in sorted(any_ofs.keys())]


def add_multiquestion_dependencies(schema, questions):
    dependencies = {}
    for key, question in questions.items():
        if question.type == 'multiquestion' and question.get('any_of'):
            dependencies.update({
                field: sorted(set(question.form_fields) - set([field]))
                for field in question.form_fields
                if len(question.form_fields) > 1
            })

    if dependencies:
        schema['dependencies'] = dependencies


def add_section(section):
    return {
        'name': section['name'],
        'editable': section['editable'],
        'optional': [optional for question in section['questions']
                     for optional in question['_optional_form_fields']],
        'required': [required for question in section['questions']
                     for required in question['required_form_fields']]
    }


def add_sections(schema, sections):
    schema['sections'] = [add_section(section) for section in sections]


def generate_schema(path, schema_type, schema_name, framework_slug, lot_slug):
    manifest = load_manifest(schema_type, framework_slug, lot_slug)
    questions = load_questions(manifest)
    drop_non_schema_questions(questions)
    schema = empty_schema(schema_name)

    build_schema_properties(schema, questions)
    add_multiquestion_anyof(schema, questions)
    add_multiquestion_dependencies(schema, questions)

    add_sections(schema, manifest.sections)

    with open(os.path.join(path, '{}-{}-{}.json'.format(schema_type, framework_slug, lot_slug)), 'w') as f:
        json.dump(schema, f, sort_keys=True, indent=2, separators=(',', ': '))
        f.write(os.linesep)
