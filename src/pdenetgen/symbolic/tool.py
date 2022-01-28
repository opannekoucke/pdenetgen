def clean_latex_name(label):
    """ Convert possible latex expression into valid variable name """

    if not isinstance(label,str):
        label = str(label)

    # -1- Supress \
    label = label.replace('\\','')

    # -2- Supress '{' .. '}'
    label = label.replace('{','')
    label = label.replace('}', '')

    # -3- Replace '^' by '_'
    label = label.replace('^','_')

    # -4- Replace ',' by '_'
    label = label.replace(',','_')

    return label
