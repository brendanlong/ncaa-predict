def list_arg(type=str, sep=",", container=list):
    def convert_arg(v):
        return container(map(type, v.split(sep)))

    return convert_arg


def team_name_to_id(name, all_teams):
    try:
        return all_teams[all_teams["school_name"] == name]["school_id"].values[0]
    except IndexError:
        raise Exception("Couldn't find ID for school [%s]" % name)
