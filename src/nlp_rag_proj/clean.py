import re
import string
from typing import Any
import pandas as pd

def normalize_text(text: str) -> str:

    # Sprawdzenie czy to string
    if not isinstance(text, str):
        return ""

    # Sprowadzenie do małych liter
    text = text.lower()

    # Usuwanie linków http lub www.
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Usuwanie znaków interpunkcyjnych
    text = re.sub(rf'[{re.escape(string.punctuation.replace('$',''))}]', '', text)

    # Usuwanie cyfr
    text = re.sub(r'[0-9]', '', text)

    # Używanie jednej spacji między słowami
    text = " ".join(text.split())

    return text
 
def apply_text_preprocessing(series: pd.Series, args: Any | None, *, stemmer, nlp_obj) -> pd.Series:
        
        if not isinstance(series, pd.Series):
            raise TypeError(f"Expected pd.Series, got: {type(series)}")
        
        series = series.apply(normalize_text)
        texts_to_process = series.tolist()

        if args.stem:
            from nlp_rag_proj.tokenization import stem
            series = stem(series, texts_to_process, stemmer)
        elif args.lem:
            from nlp_rag_proj.tokenization import lemmatize
            series = lemmatize(series, texts_to_process, nlp_obj)
        
        return series

if __name__ == "__main__":
    text = """

    When Noni Madueke was named in England's starting line-up for their World Cup opener against Croatia it was the latest moment in what has been a dramatic season for the Arsenal winger.

Last summer Madueke completed a move to the Gunners from Chelsea for about £50m, but supporters started a petition against the signing with a #NoToMadueke hashtag being used across social media platforms.

But just under 12 months later, the 24-year-old has become a Premier League winner, after helping Mikel Arteta's side to their first league title in 22 years, and is starting for Thomas Tuchel's England on the right wing.

Madueke was one of England's standout players in the 4-2 win over Croatia and won the penalty which Harry Kane scored to give the Three Lions the lead.

Madueke's Arsenal team-mate Bukayo Saka had been expected to be England's starting right winger at the World Cup but the 24-year-old is dealing with an Achilles issue he has been carrying since March.

Saka and Madueke find themselves in the unusual situation of competing for game time at both club and country.

Saka, who made his 50th appearance for England in the victory over Croatia, called the situation "unique", before adding "I don't really know how it works, but it works" when describing his relationship with Madueke.

Madueke's link with Kane and England's style
When Tuchel named his England squad for the World Cup the German was full of praise for Madueke.

The forward has put in consistently good performances for the Three Lions since Tuchel took charge and the manager said Madueke can be a "difference-maker" before highlighting his "one-on-one ability".

Tuchel has made it clear he wants his England team to play in a style that replicates the physicality of the Premier League.

And that thinking shaped the selection of his squad, with the 52-year-old picking a group of players who are physically robust and powerful runners.

Tuchel has ensured that his gameplan is built around record goalscorer and captain Kane, with the Bayern Munich forward surrounded by wingers who like to run behind the defence to leave space for him to drop deep.

That thinking worked out well with Madueke's four passes to Kane against Croatia the joint most in the England team, matched by goalkeeper Jordan Pickford.

Anthony Gordon started on the opposite flank to Madueke and their energetic performances on the wings was one of the positives of the match for Tuchel's side.

Kane is known for his passing range and when given space the striker attempted a couple of passes to release Madueke behind the Croatia defence.

Madueke had five touches in the opposition box, completed his only dribble of the game and won the penalty to set England on their way.

Despite fighting for game time, Madueke and Saka are close away from the pitch too with Saka calling his team-mate his "brother".

Arsenal boss Arteta found ways to get both wingers on the pitch at the same time in 2025-26, which may be something Tuchel opts to do as the World Cup goes on.

Madueke was utilised on the left wing by Arteta, while Saka also featured in the number 10 role as the Gunners lifted the title.

Madueke made 43 appearances last season - scoring eight goals and registering four assists in all competitions - as Arsenal ended their trophy drought.

However, Madueke only started 16 times in the league. Competing with Saka and a knee injury limited the number of times he was picked from the off.

He provided a bright spark for Arsenal in the Champions League final when replacing Saka from the bench in their loss to Paris-St Germain on penalties last month.

And he may have a similar role for his country, if England progress to the latter stages of the World Cup.

Saka continues to recover from his Achilles issue and is not expected to start until the final Group L game against Panama in New Jersey on Saturday (22:00 BST).

And with another start for Madueke likely against Ghana on Tuesday (21:00 BST), he has another chance to stake his claim that he is not just a back-up to Saka.

    """
    print(normalize_text(text))
    print(type(text))