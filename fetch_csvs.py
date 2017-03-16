#!/usr/bin/env python3
import argparse
import csv
import itertools
import multiprocessing
import os

import lxml.html
import requests


SEARCH_URL = "http://web1.ncaa.org/stats/StatsSrv/careersearch"
RECORDS_URL = "http://web1.ncaa.org/stats/exec/records"
TEAM_URL = "http://web1.ncaa.org/stats/StatsSrv/careerteam"
SCHOOL_CSV = "csv/ncaa_schools.csv"

SCRAPE_GAME_COLS = [
    "opponent_name", "game_date", "score", "opponent_score",
    "location", "neutral_site_location", "game_length",
    "attendence",
]

GAME_COLS = SCRAPE_GAME_COLS + [
    "opponent_id", "year", "school_id",
]

SCRAPE_PLAYER_COLS = [
    "player_name", "class", "season", "position", "height", "g",
    "fg_made", "fg_attempts", "fg_percent", "3pt_made",
    "3pt_attempts", "3pt_percent", "freethrows_made",
    "freethrows_attempts", "freethrows_percent", "rebounds_num",
    "rebounds_avg", "assists_num", "assists_avg", "blocks_num",
    "blocks_avg", "steals_num", "steals_avg", "points_num",
    "points_avg", "turnovers", "dd", "td",
]

PLAYER_COLS = SCRAPE_PLAYER_COLS + [
    "player_id", "year", "school_id",
]


def post_form(url, post_data=None):
    headers = {
        "user-agent":
            "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, "
            "like Gecko) Chrome/41.0.2228.0 Safari/537.36",
        "referrer": url,
    }
    if post_data is not None:
        res = requests.post(url, data=post_data, headers=headers)
    else:
        res = requests.get(url, headers=headers)
    res.raise_for_status()
    return lxml.html.document_fromstring(res.text)


def read_csv(csv_in):
    with open(csv_in, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(csv_out, data, colnames):
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    with open(csv_out, "w") as f:
        writer = csv.DictWriter(f, fieldnames=colnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def load_schools():
    if not os.path.exists(SCHOOL_CSV):
        get_schools()
    return read_csv(SCHOOL_CSV)


def get_school_games(year, school):
    print("%s/%s" % (year, school["school_name"]))
    path = "csv/games/ncaa_games_%s_%s.csv" % (year, school["school_id"])
    if not os.path.exists(path):
        int_cols = [
            "opponent_id", "score", "opponent_score", "attendence",
            "school_id",
        ]
        try:
            page = post_form(RECORDS_URL, {
                "academicYear": str(year),
                "orgId": school["school_id"],
                "sportCode": "MBB"
            })
        except requests.exceptions.HTTPError:
            # We can try again later
            print("Failed to load %s/%s" % (year, school["school_name"]))
            return []

        rows = page.xpath(
            "//form[@name='orgRecords']/table[2]/tr[position()>1]")
        games = []
        for row in rows:
            game = {
                "school_id": school["school_id"],
                "year": year,
            }
            for colname, cell in zip(SCRAPE_GAME_COLS, row.iterchildren()):
                content = cell.text_content().strip()
                if colname == "opponent_name":
                    content = content.replace("%", "").strip()
                    link = cell.xpath("a")
                    if link:
                        href = link[0].get("href")
                        _, end = href.split("(")
                        game["opponent_id"], _ = end.split(")")
                    else:
                        game["opponent_id"] = None

                if content == "-":
                    content = None
                elif colname == "attendence":
                    # remove commas
                    content = content.replace(",", "")
                game[colname] = content
            for colname in int_cols:
                if colname in game and game[colname] is not None:
                    game[colname] = int(game[colname])
            games.append(game)
        write_csv(path, games, GAME_COLS)
    return read_csv(path)


def get_games(years):
    schools = load_schools()
    for year in years:
        path = "csv/ncaa_games_%s.csv" % year
        if not os.path.exists(path):
            games = []
            for school in schools:
                games.extend(get_school_games(year, school))
            write_csv(path, games, GAME_COLS)


def get_school_players(year, school):
    print("%s/%s" % (year, school["school_name"]))
    path = "csv/players/ncaa_players_%s_%s.csv" % (year, school["school_id"])
    if not os.path.exists(path):
        int_cols = [
            "player_id", "height", "g"
        ]
        try:
            page = post_form(TEAM_URL, {
                "academicYear": str(year),
                "orgId": school["school_id"],
                "sportCode": "MBB",
                "sortOn": "0",
                "doWhat": "display",
                "playerId": "-100",
                "coachId": "-100",
                "division": "1",
                "idx": ""
            })
        except requests.exceptions.HTTPError:
            # We can try again later
            print("Failed to load %s/%s" % (year, school["school_name"]))
            return []

        rows = page.xpath(
            "//table[@class='statstable'][2]//tr[position()>3]")
        players = []
        for row in rows:
            player = {
                "school_id": school["school_id"],
                "year": year,
            }
            for colname, cell in \
                    zip(SCRAPE_PLAYER_COLS, row.iterchildren()):
                content = cell.text_content().strip()
                if colname == "player_name":
                    content = content.replace("%", "").strip()
                    link = cell.xpath("a")
                    if link:
                        href = link[0].get("href")
                        _, end = href.split("(")
                        id, _ = end.split(")")
                        if id == "-":
                            player["player_id"] = None
                        else:
                            player["player_id"] = id
                    else:
                        player["player_id"] = None

                if content == "-":
                    content = None
                elif colname == "height":
                    feet, inches = content.split("-")
                    content = int(feet) * 12 + int(inches)
                player[colname] = content
            for colname in int_cols:
                if player[colname] is not None:
                    player[colname] = int(player[colname])
            if player["player_name"] is not None:
                players.append(player)
        write_csv(path, players, PLAYER_COLS)
    return read_csv(path)


def get_players(years):
    schools = load_schools()
    for year in years:
        path = "csv/ncaa_players_%s.csv" % year
        if not os.path.exists(path):
            players = []
            for school in schools:
                players.extend(get_school_players(year, school))
            write_csv(path, players, PLAYER_COLS)


def get_schools():
    page = post_form(SEARCH_URL)
    options = page.xpath("//select[@name='searchOrg']/option[position()>1]")
    schools = [
        {"school_id": option.get("value"), "school_name": option.text}
        for option in options]
    write_csv(SCHOOL_CSV, schools, colnames=list(schools[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_name")
    subparsers.required = True
    commands = [
        ("get_games", get_games),
        ("get_players", get_players),
        ("get_schools", get_schools),
    ]
    for name, func in commands:
        subparser = subparsers.add_parser(name)
        subparser.set_defaults(func=func)
        if func != get_schools:
            subparser.add_argument(
                "--years", "-y", type=lambda v: map(int, v.split(",")),
                default=list(range(2002, 2018)),
                help="The years to scrape data for. (default: %(default)s")
    args = parser.parse_args()
    if args.func == get_schools:
        args.func()
    else:
        args.func(args.years)
