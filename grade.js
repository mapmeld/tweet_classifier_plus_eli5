// security.csp.enable = false (Firefox)

// new twitter
let replies = [],
    oldTwitter = true;

if (oldTwitter) {
  replies = $('.PermalinkOverlay-modal .tweet .tweet-text')
} else {
  replies = document.querySelectorAll('article [data-testid="tweet"] [lang="en"] span, article [data-testid="tweet"] [lang="und"] span');
}

function workOnReply(index) {
  let tweetText = replies[index].innerText.toLowerCase();

  fetch("http://localhost:5000/tweet", {
    method: 'POST',
    body: tweetText
  }).then(res => res.text()).then((readout) => {
    let lines = readout.split("\n");

    let category = lines[1];
    let totalScore = category.split('=')[2].split(')')[0] * 1;
    if (category.indexOf('less weird') > -1) {
      // less weird
      category = 'less weird';
    } else {
      // known weird
      category = 'known weird';
    }

    let contributingWords = lines.slice(4);
    contributingWords.pop();

    tweetText = ' ' + tweetText + ' ';
    tweetText = tweetText.replace(/\,/, ' , ');
    tweetText = tweetText.replace(/\./, ' . ');
    tweetText = tweetText.replace(/\?/, ' ? ');
    tweetText = tweetText.replace(/\!/, ' ! ');

    contributingWords.forEach((line) => {
      let contribution = line.trim().split(/\s+/);
      let score = contribution[0] * 1;
      let word = contribution[1];

      if (word === '<BIAS>') {
        return;
      }

      if (category === 'known weird') {
        // flip score for consistency
        score *= -1;
      }
      let color = (score < 0) ? 0 : 120;
      let percent = Math.round((1 - 5 * Math.abs(score)) * 100);

      let rg = new RegExp('\\s' + word + '\\s', 'i');
      if (rg.test(tweetText)) {
        tweetText = tweetText.replace(rg, ' <span style="opacity: 0.8; background-color: hsl(' + color + ', 100.00%, ' + percent + '.00%)">' + word + '</span> ')
      } else {
        //alert(word);
      }
    });

    if (oldTwitter) {
      $('.PermalinkOverlay-modal .tweet .tweet-text')[index].innerHTML = tweetText;
    }

    // rgb
    // red: background-color: hsl(0, 100.00%, 93.27%); opacity: 0.8-1.0
    // green: background-color: hsl(120, 100.00%, 93.56%); opacity: 0.81
    // final % is 60-98% with lower % = darker = stronger score

    if (index + 1 < replies.length) {
      workOnReply(index + 1);
    }
  });
}
workOnReply(oldTwitter ? 1 : 0);
