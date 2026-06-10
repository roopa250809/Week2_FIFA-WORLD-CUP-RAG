import os
import re
import pandas as pd
from typing import List
from langchain_core.documents import Document


DATA_DIR = "data"


def _normalize_team(name: str) -> str:
    """Keep historical team names but make them consistent."""
    if not isinstance(name, str):
        return str(name)
    name = name.strip()
    return name


def load_tournament_standings() -> List[Document]:
    """
    One CSV per tournament (FIFA - 1930.csv ... FIFA - 2022.csv).
    Each row = one team's performance that year → one Document.
    """
    docs = []
    pattern = re.compile(r"FIFA - (\d{4})\.csv$")

    for fname in sorted(os.listdir(DATA_DIR)):
        match = pattern.match(fname)
        if not match:
            continue
        year = int(match.group(1))
        df = pd.read_csv(os.path.join(DATA_DIR, fname))

        for _, row in df.iterrows():
            team = _normalize_team(row["Team"])
            position = int(row["Position"])
            gp = int(row["Games Played"])
            w  = int(row["Win"])
            d  = int(row["Draw"])
            l  = int(row["Loss"])
            gf = int(row["Goals For"])
            ga = int(row["Goals Against"])
            gd = row["Goal Difference"]
            pts = int(row["Points"])

            text = (
                f"In the {year} FIFA World Cup, {team} finished in position {position}. "
                f"{team} played {gp} matches, with {w} wins, {d} draws, and {l} losses. "
                f"They scored {gf} goals and conceded {ga} goals "
                f"(goal difference: {gd}), earning {pts} points in the tournament."
            )

            docs.append(Document(
                page_content=text,
                metadata={
                    "source": "tournament_standings",
                    "year": year,
                    "team": team,
                    "position": position,
                    "doc_id": f"standings-{year}-{team}",
                }
            ))
    return docs


def load_world_cup_summary() -> List[Document]:
    """
    FIFA - World Cup Summary.csv: one row per tournament (1930-2022).
    Each row = one tournament overview document.
    """
    path = os.path.join(DATA_DIR, "FIFA - World Cup Summary.csv")
    df = pd.read_csv(path)
    docs = []

    for _, row in df.iterrows():
        year = int(row["YEAR"])
        host = row["HOST"]
        champion = row["CHAMPION"]
        runner_up = row["RUNNER UP"]
        third = row["THIRD PLACE"]
        teams = int(row["TEAMS"])
        matches = int(row["MATCHES PLAYED"])
        goals = int(row["GOALS SCORED"])
        avg = row["AVG GOALS PER GAME"]

        text = (
            f"The {year} FIFA World Cup was hosted by {host}. "
            f"{champion} won the tournament, defeating {runner_up} in the final. "
            f"{third} finished in third place. "
            f"A total of {teams} teams competed, playing {matches} matches "
            f"and scoring {goals} goals overall (average of {avg} goals per game)."
        )

        docs.append(Document(
            page_content=text,
            metadata={
                "source": "tournament_summary",
                "year": year,
                "host": host,
                "champion": champion,
                "runner_up": runner_up,
                "third_place": third,
                "doc_id": f"summary-{year}",
            }
        ))
    return docs


def load_analytical_insights() -> List[Document]:
    """
    Pre-computed cross-tournament analytical documents that answer
    aggregation queries no single standing/summary doc can answer.
    """
    summary = pd.read_csv(os.path.join(DATA_DIR, "FIFA - World Cup Summary.csv"))
    champions = dict(zip(summary["YEAR"], summary["CHAMPION"]))
    pattern = re.compile(r"FIFA - (\d{4})\.csv$")

    records = []
    for fname in sorted(os.listdir(DATA_DIR)):
        m = pattern.match(fname)
        if not m:
            continue
        year = int(m.group(1))
        champ = champions.get(year)
        df = pd.read_csv(os.path.join(DATA_DIR, fname))
        row = df[df["Team"] == champ]
        if row.empty:
            continue
        row = row.iloc[0]
        records.append({
            "year": year,
            "team": champ,
            "ga":   int(row["Goals Against"]),
            "gf":   int(row["Goals For"]),
            "gp":   int(row["Games Played"]),
            "wins": int(row["Win"]),
        })

    by_ga = sorted(records, key=lambda x: (x["ga"], x["ga"] / x["gp"]))

    lines = ["Defensive records of FIFA World Cup winning teams (goals conceded during their winning campaign):"]
    for r in by_ga:
        ga_per_game = round(r["ga"] / r["gp"], 2)
        lines.append(
            f"{r['year']} - {r['team']}: {r['ga']} goals conceded in {r['gp']} games "
            f"({ga_per_game} goals against per game), scored {r['gf']} goals."
        )
    top3 = by_ga[:3]
    lines.append(
        f"The best defensive records among World Cup winners are: "
        f"{top3[0]['team']} ({top3[0]['year']}) with {top3[0]['ga']} goals conceded, "
        f"{top3[1]['team']} ({top3[1]['year']}) with {top3[1]['ga']} goals conceded, and "
        f"{top3[2]['team']} ({top3[2]['year']}) with {top3[2]['ga']} goals conceded — "
        f"all in 7 games. France 1998, Italy 2006, and Spain 2010 share the record for "
        f"fewest goals conceded (2) in a winning World Cup campaign."
    )

    doc_text = "\n".join(lines)
    docs = [Document(
        page_content=doc_text,
        metadata={
            "source": "analytical_insight",
            "topic":  "champion_defensive_records",
            "doc_id": "insight-champion-defense",
        }
    )]

    # Also add goals scored ranking
    by_gf = sorted(records, key=lambda x: x["gf"], reverse=True)
    gf_lines = ["Goals scored by FIFA World Cup winning teams during their winning campaign:"]
    for r in by_gf:
        gf_lines.append(
            f"{r['year']} - {r['team']}: {r['gf']} goals scored in {r['gp']} games "
            f"({round(r['gf']/r['gp'], 2)} per game), conceded {r['ga']}."
        )
    docs.append(Document(
        page_content="\n".join(gf_lines),
        metadata={
            "source": "analytical_insight",
            "topic":  "champion_offensive_records",
            "doc_id": "insight-champion-offense",
        }
    ))

    return docs


def load_player_insights() -> List[Document]:
    """
    Hard-coded player award and top-scorer facts that are not derivable
    from team-level CSVs but are well-known World Cup records.
    """
    docs = []

    # Top scorers per tournament (Golden Boot winners)
    top_scorers = [
        (1930, "Guillermo Stábile", "Argentina", 8),
        (1934, "Oldrich Nejedly", "Czechoslovakia", 5),
        (1938, "Leônidas", "Brazil", 7),
        (1950, "Ademir", "Brazil", 9),
        (1954, "Sándor Kocsis", "Hungary", 11),
        (1958, "Just Fontaine", "France", 13),
        (1962, "Florian Albert / Garrincha / Vavá / others", "various", 4),
        (1966, "Eusébio", "Portugal", 9),
        (1970, "Gerd Müller", "West Germany", 10),
        (1974, "Grzegorz Lato", "Poland", 7),
        (1978, "Mario Kempes", "Argentina", 6),
        (1982, "Paolo Rossi", "Italy", 6),
        (1986, "Gary Lineker", "England", 6),
        (1990, "Salvatore Schillaci", "Italy", 6),
        (1994, "Hristo Stoichkov / Oleg Salenko", "Bulgaria/Russia", 6),
        (1998, "Davor Šuker", "Croatia", 6),
        (2002, "Ronaldo", "Brazil", 8),
        (2006, "Miroslav Klose", "Germany", 5),
        (2010, "Thomas Müller / David Villa / Wesley Sneijder / Diego Forlán", "various", 5),
        (2014, "James Rodríguez", "Colombia", 6),
        (2018, "Harry Kane", "England", 6),
        (2022, "Kylian Mbappé", "France", 8),
    ]

    lines = ["FIFA World Cup Golden Boot (top scorer) winners by tournament:"]
    for year, player, nation, goals in top_scorers:
        lines.append(f"{year}: {player} ({nation}) — {goals} goals")
    lines.append(
        "All-time record: Just Fontaine (France) scored 13 goals in 1958, "
        "the most by any player in a single World Cup tournament. "
        "Ronaldo (Brazil) scored 15 World Cup goals total across 1998, 2002, and 2006, "
        "the all-time World Cup goals record until Miroslav Klose surpassed him with 16 goals."
    )
    docs.append(Document(
        page_content="\n".join(lines),
        metadata={
            "source": "analytical_insight",
            "topic":  "golden_boot_winners",
            "doc_id": "insight-golden-boot",
        }
    ))

    # 2022 World Cup individual awards and key player stats
    awards_2022 = """2022 FIFA World Cup individual awards and key player performances:
- Golden Ball (best player): Lionel Messi (Argentina) — scored 7 goals and provided 3 assists in 7 matches, the most goal contributions of any player in the tournament.
- Golden Boot (top scorer): Kylian Mbappé (France) — scored 8 goals in 7 matches, including a hat-trick in the final against Argentina.
- Golden Glove (best goalkeeper): Emiliano Martínez (Argentina).
- Best Young Player: Enzo Fernández (Argentina).
- Lionel Messi scored 7 goals in the 2022 FIFA World Cup: vs Saudi Arabia (1), vs Mexico (1), vs Australia (1), vs Netherlands (1, pen), vs Croatia (2, including 1 pen), vs France (2, including 1 pen in the final).
- Julián Álvarez (Argentina) scored 4 goals in the 2022 World Cup.
- Argentina scored 15 goals total in the 2022 World Cup across 7 matches.
- France's Kylian Mbappé scored 8 goals — the most by any player in the 2022 tournament.
- Cristiano Ronaldo (Portugal) scored 1 goal in the 2022 FIFA World Cup — a penalty against Ghana in the group stage. With that goal he became the first man to score in 5 different FIFA World Cups (2006, 2010, 2014, 2018, 2022). Portugal were eliminated in the quarterfinals by Morocco.
- Harry Kane (England) scored 0 goals in the 2022 World Cup; England were eliminated in the quarterfinals by France."""
    docs.append(Document(
        page_content=awards_2022,
        metadata={
            "source": "analytical_insight",
            "topic":  "2022_world_cup_awards",
            "doc_id": "insight-2022-awards",
        }
    ))

    return docs


def build_corpus():
    """Combine all sources into one document list."""
    from src.ingest_wiki import load_wikipedia_articles

    standings = load_tournament_standings()
    summaries = load_world_cup_summary()
    wiki_docs = load_wikipedia_articles()
    insights  = load_analytical_insights()
    player_insights = load_player_insights()

    all_docs = standings + summaries + wiki_docs + insights + player_insights
    print(f"\n--- Corpus stats ---")
    print(f"Standings docs (row-level chunks):  {len(standings)}")
    print(f"Tournament summary docs:            {len(summaries)}")
    print(f"Wikipedia chunks (semantic split):  {len(wiki_docs)}")
    print(f"Analytical insight docs:            {len(insights)}")
    print(f"Player insight docs:                {len(player_insights)}")
    print(f"Total documents in corpus:          {len(all_docs)}")
    return all_docs


if __name__ == "__main__":
    docs = build_corpus()
    print("\n--- Sample standings doc ---")
    print(docs[0].page_content)
    print("\n--- Sample Wikipedia doc ---")
    wiki_sample = next(d for d in docs if d.metadata["source"] == "wikipedia")
    print(wiki_sample.page_content[:300], "...")
    print("Metadata:", wiki_sample.metadata)




