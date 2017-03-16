This data is scraped from NCAA's stats tool on various pages.

Beware that random columns are null, which is indicated by `-` in
the HTML. Sometimes there is no data for a team, but there will be
a blank row instead of no rows.

Only make one request at a time or NCAA will temporarily block you.

## `fetch_csvs.py`

See [`fetch_csvs.py`](../fetch_csvs.py) to do this automatically. Run with
`--help` for additional arguments like the years to download.

We also do some cleanup:

  - Replace '-' with empty string
  - Remove blank rows
  - Convert height from feet-inches to just inches
  - Remove commas in attendence numbers

## NCAA API Info

### School Info

  - Command: `./fetch_csvs.py get_schools`
  - URL: <http://web1.ncaa.org/stats/StatsSrv/careersearch>
  - How: Just navigate to that page normally.
  - Elements: The "School" dropdown contains names as text and ID's
    as values. Note that the first `<option>` is "All", so make
    sure to skip that.
  - XPath: `//select[@name='searchOrg']/option[position()>1]`

### Player Info

  - Command: `./fetch_csvs.py get_players`
  - URL: <http://web1.ncaa.org/stats/StatsSrv/careerteam>
  - How: From the search page in School Info, select your school,
    year, and sport, click "Search", then click on the school's
    name in the bottom table.
      - As a script, POST this data to the URL:
          - sortOn: 0
          - doWhat: "display"
          - playerId: -100
          - coachId: -100
          - orgId: The school's ID
          - academicYear: The year (for 2015-16, send "2016")
          - division: 1
          - sportCode: "MBB"
           - idx: ""
  - Elements: The table of player statistics at the bottom. Note
    that we can get the player ID from the link href in the first
    column. Make sure to skip the headers.
  - XPath: `//table[@class='statstable'][2]//tr[position()>3]`

### Game Info

  - Command: `./fetch_csvs.py get_games`
  - URL: <http://web1.ncaa.org/stats/exec/records>
  - How: From the Player Info page, click "View [Team Name]
    Year-By-Year W/L Record", then click a row in the "W-L" column.
      - As a script, POST this data to the URL:
          - academicYear: The year (for 2015-16, send "2016")
          - orgId: The school ID
          - sportCode: "MBB"
  - Elements: The second table. Note that we can get the opponent's
    ID from the link href in the first column.
  - XPath: `//form[@name='orgRecords']/table[2]/tr[position()>1]`
