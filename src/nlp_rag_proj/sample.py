from nlp_rag_proj.io import load_model
from sklearn.pipeline import Pipeline
from pathlib import Path

def predict_sample(model: Pipeline, list_text: list[str]) -> list[str]:

    if isinstance(list_text, str):
        list_text = [list_text]

    if not list_text and not all(isinstance(element, str) for element in list_text):
        raise TypeError(f"all elements in list_text must be {type(str)}")
    
    prediction = model.predict(list_text)
    return prediction

if __name__ == "__main__":
    text_sport = """

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
    text_politics = """
Sir Keir Starmer has said he will quit as Labour Party leader, paving the way for a contest to decide a new prime minister.

Speaking in Downing Street, Sir Keir said he accepted he was not best placed to lead Labour into the next general election and he had informed the King of his decision to step down.

Sir Keir added he has asked Labour's governing body to set out a timetable to replace him, with nominations opening on 9 July and ending by the summer recess on 16 July.

He said if there was a contest then a new leader would be in place before Parliament returns in September, and he will "do everything" he can to ensure an "orderly" transition of power.

Sir Keir said he would remain as prime minister until the leadership contest is complete.

He added he would also give his successor "my full and unequivocal support, knowing that they will inherit a Britain that is far stronger and fairer than the one I inherited two years ago".

Andy Burnham is regarded by many as the frontrunner to replace Sir Keir after he secured an emphatic win over his Reform UK rival in last week's Makerfield by-election.

Burnham announced on Monday that he would put himself forward as a candidate in the leadership contest, before he boarded a train to London to take his parliamentary seat.

His chances were given an immediate boost by former Health Secretary Wes Streeting, who had been viewed as his main rival, offering his backing to the former Greater Manchester mayor.

Speaking to the BBC as he arrived at Euston station, Burnham praised Sir Keir's "dedication and service".

Asked if he would call a general election in the event that he became prime minister, he replied: "You're jumping several hurdles ahead. My priority is to be sworn in as the MP for Makerfield."

On being formally sworn in as an MP in the House of Commons, Burham was greeted by loud cheers from Labour benches and a few heckles from the opposition, with one MP shouting: "He's not the messiah."

After taking his seat, he joined around 200 Labour MPs in Westminster Hall to take a group selfie.

PA Media Andy Burnham in Westminster Hall with his arm raised up to take a selfie of himself and the group of around 200 Labour MPs standing behind him. The MPs standing immediately next to him include Nia Griffith, Kate Osborne and Paulette HamiltonPA Media
Sir Keir was elected leader of the Labour Party in April 2020 and became prime minister on 5 July 2024 following Labour's landslide general election victory.

He will leave Downing Street as the shortest-serving Labour prime minister in history.

His period in office will last longer than his Conservative predecessors Rishi Sunak and Liz Truss but behind all six previous Labour prime ministers.

Sir Keir's decision to step down also means the UK will soon have its seventh prime minister since 2016.

Speaking at a lectern in Downing Street, Sir Keir said his party had asked "whether I am best placed to lead us into the next general election".

He said: "I have heard the answer of my parliamentary party to that question, and I accept that answer with good grace."

Sir Keir was accompanied by his wife, Victoria, as he walked out to deliver his resignation speech at 09:30 BST in the blazing sunshine.

Watched by his supporters, colleagues and No 10 staff, Sir Keir's voice cracked with emotion as he spoke of what his focus will be on next.

He said: "When I leave the biggest job in the country, I shall spend more time on the most important job: being the best husband I can to my fantastic wife Vic, who has been a rock by my side through good times and bad; and being the best dad I can to my beautiful children, who are my pride and my joy."

The sound of Beethoven's Ode to Joy could be heard playing in the background as Sir Keir delivered his speech, with the EU anthem being played by a protester.

Sir Keir once described it as the piece of music that best "sums up" his party, telling Classic FM in 2023 that the symphony had a "sense of destiny and is hugely optimistic... it's that sense of moving forward to a better place".

Chancellor Rachel Reeves paid tribute to Sir Keir for helping to "build a stronger, more secure Britain", saying the pair had "achieved a lot together to be proud of, and there is more to do".

Former Labour deputy leader Angela Rayner said "history will remember not just the challenges he faced but the achievements he oversaw", as she pointed to reforms to employment and leasehold legislation.

Analysis: Everything points to Burnham becoming PM within weeks
Why did Keir Starmer resign and what could happen next?
Prime Minister Keir Starmer's resignation speech in full
Burnham thanked Sir Keir for his leadership and said the country now expects "stability, seriousness and a continued focus on the issues that matter most and that is what it will get".

Announcing his widely-expected decision to stand in the leadership contest, he wrote on X: "People want to see progress on economic growth, cost of living, public services, housing and opportunities for the next generation."

Streeting had previously outlined his intention to join any Labour leadership contest", but on Monday said he had "spoken at length with Andy in recent days" and called on colleagues to back Burnham.

Streeting said he was convinced that Burnham "is committed to building an inclusive party that draws on the best of our political traditions" and that he "can win the fight of our lives against the force of nationalism".

Sir Keir had spent the weekend mulling over his future at Chequers, the prime minister's country residence in Buckinghamshire.

Pressure from within Labour had been mounting on Sir Keir to outline a timetable for his departure following Burnham's victory in the Makerfield by-election.

Discontent towards Sir Keir's leadership had also been rising before a poor set of election results in England, Wales and Scotland in May.

This included over his his decision to change direction on three major policies in a month after pressure from within his own party.

Sir Keir's decision to appoint Lord Mandelson as UK ambassador to the US also led to questions about his judgement and the wider Downing Street operation.

Lord Mandelson was sacked after new information came to light about the depth of his relationship with the late convicted sex offender Jeffrey Epstein.

Reuters Sir Keir Starmer and Lady Victoria Starmer are stood in front of a black door with the number 10 written on it. Sir Keir is wearing a dark-coloured suit jacket, white shirt, patterned tie and glasses. Lady Starmer is wearing a white top. Sir Keir is looking ahead and Lady Starmer is looking at her husband.Reuters
Sir Keir Starmer described Lady Victoria Starmer as a 'rock by my side'
Sir Keir opened his resignation speech by defending his record in government, including on employment rights, immigration and child poverty.

He also argued that he had changed Labour after inheriting a party that was "politically, financially and morally bankrupt".

Paying tribute to his outgoing leader, Deputy Prime Minister David Lammy said Sir Keir's record on foreign policy was "second to none".

He told MPs in the House of Commons that the prime minister had been "principled, courageous and on the right side of history".

Conservative Party leader Kemi Badenoch described Sir Keir as a "terrible prime minister" and attacked his policies, including the rise in employer National Insurance contributions and "giving up on real welfare reform".

She wrote on X: "But the problem isn't just Starmer.

"Labour MPs only want higher taxes to hand out more benefits, as the welfare secretary has pointed out. These are Labour's choices and their values, regardless of who is running the party."

Liberal Democrat leader Sir Ed Davey said the British people were "sick of being let down by an endless merry-go-round of prime ministers while nothing really changes".

He said: "This time must be different. It can't just be about changing who's in Number 10, it has to be about changing our broken politics so we can fix our country."

Reform leader Nigel Farage demanded a general election, saying: "If Labour thinks it can shove another professional politician into No 10, it has another thing coming."

Zack Polanski, leader of the Green Party of England and Wales, said people would hope that Burnham can bring about "meaningful change" but the "jury is out".

"The question for Burnham is - are you willing to tax wealth fairly, are you willing to bring our water companies into public ownership, are you willing to bring in PR [proportional representation] so we can have a fair voting system and a better politics that represents everyone."


    """
    text_entertainment = """
Scroll through TikTok for long enough and you'll find short films about fictional relationships, with painstaking fan edits, imagined future storylines, AI-generated posters and millions of comments debating them as if they're real.

This level of fandom has transformed young adult romance, often found in novels, into one of streaming's safest bets.

With #BookTok attracting around 80 million posts on TikTok and #romancebooks more than 6.5 million, platforms are now investing heavily in bestselling novels with passionate online audiences - before a single scene is filmed.

The latest entry into that booming genre is Your Fault: London.

A film based on the bestselling novels by Spanish author Mercedes Ron, it stars Asha Banks and Matthew Broome, who say they have been struck by the intensity of the franchise's existing fanbase.

Getty Images Mercedes Ron in evening dress and a diamond choker - she is young with long, mid-brown hair Getty Images
Spanish writer Mercedes Ron, 33, is behind the Culpables Saga novels - a trilogy called My Fault, Your Fault and Our Fault
"The fans are the reason the films are so successful," Banks tells the BBC. "The visibility of it and where it lives is so much on social media."

This film is the latest chapter in an unlikely success story that began not in Hollywood, but on the reading platform Wattpad, where the Argentine-born Spanish writer Ron first published the Culpables trilogy.

The books - called Culpa Mia, Culpa Tuya and Culpa Nuestra were all written before she was aged 20, and after launching on Wattpad, they went on to became bestsellers, and hugely popular with teenagers.

They were then adapted into a hugely successful series of Spanish-language films, simply titled Your Fault.

The English-language version of the story - Your Fault: London, story follows Noah, who moves to the capital city after her mother marries a wealthy businessman.

Noah falls for the businessman's son, Nick. The step-siblings' romance is tested by family tensions, buried secrets and their own impulsive decisions.

Nicole Clemens, head of UK and international originals at Amazon MGM Studios, says the Culpables films have collectively reached 100 million viewers worldwide.

The films were number one in more than 170 countries at launch, with over 90% of viewers coming from outside Spain, Amazon said in February.

Amazon Matthew Broome as Nick kisses Noah's neck as they sit on a bed Amazon
Nick and Noah's romantic journey is explored in the film
Your Fault: London, follows on from 2025's My Fault: London, and will also be followed by Our Fault: London.

The middle film picks up as Noah starts Oxford University and Nick focuses on his career, with new relationships and old conflicts threatening to pull them apart.

Banks promises "a lot of drama", while Broome says audiences will constantly be torn between the pair.

"You'll be agreeing with one character and then think, 'Oh no, the other one is right.'"

The English-language version has a new cast and a London backdrop.

Banks, 22, thought "it was interesting they were doing an English version so quickly".

"But ultimately people love the story and resonated with it so much and that's why it was such a success. So why not?

"It's so much fun, it's addictive and entertaining and it's great to bring it to a new audience."

Broome agrees and says the rapid turnaround "emphasises the power of Mercedes Ron and her books".

'People will definitely watch it'
For the actors, the most unusual part of the experience has been joining a story that already had millions of passionate readers and viewers.

"You have mega fans of all of these BookTok books," Broome, 25, says.

"You're making something with an added level of excitement because you know people are going to watch it. We don't know how they'll receive it, but they will definitely watch it."

Some critics have called Your Fault: London clichéd, unnecessary and "fanfiction-like" - Variety said the first film in the trilogy was "a trashy gimmick" and a "tasteless" adaptation of "tawdry teen literature".

Similarly, The Guardian called the original Spanish films "a bizarre and wooden step-sibling romance".

But fans of the books and Spanish films have embraced it.

BookTok creator Tia Saunders, 22, says the films are "phenomenal and amazing adaptation".

"I love the forbidden love element and the casting is brilliant."

Amazon Asha Banks as Noah sits on a sofa in a garden wearing sunglasses and a red hoodieAmazon
Banks says she hopes fans will like the way she's portrayed Noah
Broome and Banks are grateful for the fans but admit that sometimes the online attention can also be overwhelming.

Broome says he regularly comes across AI-generated images imagining the pair's future together "which is kind of crazy".

"I'll see a poster and I think, 'Wow, we've released a poster' but it's actually just a fan creating one."

Having each other has helped the young actors navigate the attention, and Broome explains that "as soon we're overwhelmed we just talk to each other".

He adds that because much of the fandom exists online, "if you put your phone down you can disconnect from it".

The pair deliberately avoided watching the Spanish version, wanting to develop their own interpretations of the characters.

They only watched the originals after filming had wrapped and "it was lovely because we watched it together and our version was done so we couldn't compare," Banks says.

Broome believes the British adaptation stands apart because "our dynamic is completely different" and Banks adds that the change in setting gives the story a new identity.

'Easy to romanticise toxicity'
The series has sparked debate over whether it romanticises toxic relationships, with Noah and Nick's romance often marked by jealousy, secrecy and conflict.

Broome argues the films instead show the consequences of those behaviours.

"You see it go wrong with Noah and Nick and you see the bad toxic traits that fall out of that. Then you see what you need to resolve the conflict, which is healthy communication."

Banks says they were conscious of portraying two young people still figuring themselves out, rather than presenting their relationship as an ideal.

"It's so easy to romanticise toxicity, and their relationship is full of that, but you are watching them work that out themselves alongside the audience," she says.

The actress adds that one of the aspects she most enjoyed was Noah's resilience, saying "it was so lovely for me to play a strong young female character and I hope that's reflected on the screen".
    """
    text_random = "mleko"
    model: Pipeline = load_model(Path.cwd() / "models" / "tfidf_svc_nl.joblib")
    print(predict_sample(model, [text_sport, text_politics, text_entertainment, text_random]))
    print(predict_sample(model, text_sport))
    #print(predict_sample(model, 5))
