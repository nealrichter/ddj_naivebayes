use strict;
use DB_File;

# Dr. Dobb's Journal
# May 01, 2005
# Naive Bayesian Text Classification
# Fast, accurate, and easy to implement
# John Graham-Cumming
# http://www.ddj.com/development-tools/184406064
# 
# Extended by Neal Richter
#  - parse CSV text and treat phrases as intact symbols
#  - export the model to CSV file
#  - print stats on the model
#  - prune the model
# 
# Neal's notes
# To Train:
# find label1/training/ -exec perl naivebayes.pl add label1 '{}' \;
# find label2/training/ -exec perl naivebayes.pl add label2 '{}' \;
# 
# To Test:
# perl naivebayes.pl classify label1/testing/somedatafile
# perl naivebayes.pl classify label2/testing/somedatafile
# The label with the smallest number wins (ie the first one in the list)
# 
# Paying attention to the absolute difference between the scores is important as well.  See the Naive Bayes literature for details.
# 
# To Prune - removes words in the model with less than X frequency:
# perl naivebayes.pl prune 10
# 
# To show stats:
# perl naivebayes.pl stats
# 
# To export the model to a CSV file
# perl naivebayes.pl export <file>
# 
# Hash with two levels of keys: $words{category}{word} gives count of
# 'word' in 'category'.  Tied to a DB_File to keep it persistent.
# 

my %words;
tie %words, 'DB_File', 'words.db';

#Utils
sub trim {
  my $string = shift;
  for ($string) {
    s/^\s+//;
    s/\s+$//;
    s/=/_/g;
    s/-/_/g;
  }
  return $string;
}

# Read a file and return a hash of the word counts in that file
sub parse_file
{
    my ( $file ) = @_;
    my %word_counts;


    open FILE, "<$file";
    while ( my $line = <FILE> ) {
#        print "parsing $line\n";

#       split on CSV
        my @values = split(/[,;:]/, $line);   #replace with this for whitespace /[ \t\n\r,;:]/
                                              #Prior pattern  while ( $line =~ s/([[:alpha:#]]{3,44})[ \t\n\r,;:]// ) {

#       Grab all the words with between 3 and 44 letters
        foreach my $val (@values) {
            my $wrd = lc(trim($val));
            if ((length( $wrd) > 2) && (length( $wrd) < 45) && ($wrd =~ m/[A-Za-z0-9_#]/)) {
               $word_counts{lc($wrd)}++;
            }
#            else { print "Rejected: $wrd \n"; }
        }
    }
    close FILE;
    return %word_counts;
}

# Add words from a hash to the word counts for a category
sub add_words
{
    my ( $category, %words_in_file ) = @_;

    foreach my $word (keys %words_in_file) {
        $words{"$category=$word"} += $words_in_file{$word};
    }
}

# Get the classification of a file from word counts
sub classify
{
    my ( %words_in_file ) = @_;

    # Calculate the total number of words in each category and
    # the total number of words overall

    my %count;
    my $total = 0;
    foreach my $entry (keys %words) {
        $entry =~ /^(.+)=(.+)$/; #category=word
        $count{$1} += $words{$entry};
        $total += $words{$entry};
    }

    # Run through words and calculate the probability for each category

    my %score;
    foreach my $word (sort (keys(%words_in_file))) {
        foreach my $category (keys %count) {
            if ( defined( $words{"$category=$word"} ) ) {
                #print "[$word] $category: ".$score{$category}." += log(". $words{"$category=$word"}." / ".$count{$category}." )\n";
                $score{$category} += log( $words{"$category=$word"} /
                                          $count{$category} );
            } else {
                #print "[$word] $category: ".$score{$category}." += log( 0.01 / ".$count{$category}." )\n";
                $score{$category} += log( 0.01 /
                                          $count{$category} );
            }
        }
    }
    # Add in the probability that the text is of a specific category

    foreach my $category (keys %count) {
        #print "(pr) $category: ".$score{$category}." += log(". $count{$category}." / ".$total." )\n";
        $score{$category} += log( $count{$category} / $total );
    }

    #print the log likelyhood of the categories in sorted order
    my @score_array;
    my @class_array;

    foreach my $category (sort { $score{$b} <=> $score{$a} } keys %count) {
        print "$category $score{$category}\n";
        $score_array[@score_array] = $score{$category}; 
        $class_array[@class_array] = $category; 
    }
 
    my $sz = scalar @score_array;
    if($sz > 1)
    {
      for (my $count=1; $count<$sz; $count++)
      {
        print "Test: of ". $class_array[$count-1] . " vs " . $class_array[$count] . " = " . (2 * ($score_array[$count-1] - $score_array[$count])) . "\n";
        print "Confidence = " . (1- abs((2 * ($score_array[$count-1] - $score_array[$count]))/$score_array[$count-1])) . " - ". abs((2 * ($score_array[$count-1] - $score_array[$count]))/$score_array[$count-1])."\n";
      } 
    }

}

# Get the classification of a file from word counts
sub prune_model
{
    my ($thresh) = @_;

    # Calculate the total number of words in each category and
    # the total number of words overall

    my %word_freq;
    my $total = 0;
    my $category;
    my $word;
    foreach my $entry (keys %words) {
        $entry =~ /^(.+)=(.+)$/;
        $category = $1;
        $word = $2;
        $word_freq{$word} += $words{$entry};
        $total += $words{$entry};
    }

    # Run through words and calculate the probability for each category
    # Add to new pruned model if greater than the threshold

    unlink 'words_pruned.db';
    my %words_pruned;
    tie %words_pruned, 'DB_File', 'words_pruned.db';

    print "total words:$total\n";

    foreach my $entry (keys %words) {
        $entry =~ /^(.+)=(.+)$/;  #category=word
        $category = $1;
        $word = $2;
        if( $word_freq{$word} >= $thresh ) {  #freq based threshold
            $words_pruned{$entry} = $words{$entry};
        }  
        else  
        { 
           #print "Skipped: [$entry],$words{$entry} ".$word_freq{$word}." ; $thresh\n"; 
        }
    }

    untie %words_pruned;
}

# Get the classification of a file from word counts
sub model_stats
{
    # Calculate the total number of words in each category, total categories
    # the total number of words overall

    my %categories;
    my %category_uniques;
    my $unique_words;
    my $total = 0;
    foreach my $entry (keys %words) {
        $entry =~ /^(.+)=(.+)$/; #category=word
        $categories{$1} += $words{$entry};
        $category_uniques{$1}++;
        $total += $words{$entry};
        $unique_words++;
    }

    print "Total words: $total\n";
    print "Unique words: $unique_words\n";
    print "Categories:\n";
    foreach my $entry (keys %categories) {
        print " - $entry: " . $categories{$entry} . " words, ". $category_uniques{$entry} . " unique words\n";
    }
}

# Export a CSV of the model
sub model_export
{
    my ($export_file) = @_;
    print "Opening: $export_file\n";
    open(EXPCSV, ">$export_file");

    # Calculate the total number of words in each category, total categories
    # the total number of words overall

    my %categories;
    my %category_uniques;
    my $unique_words;
    my $total = 0;
    my $category;
    my $word;
    my $count;
    foreach my $entry (keys %words) {
        $entry =~ /^(.+)=(.+)$/; #category=word
        $category = $1;
        $word = $2;
        $categories{$1} += $words{$entry};
        $category_uniques{$1}++;
        $total += $words{$entry};
        $unique_words++;
    }

    print EXPCSV "#Total words: $total\n";
    print EXPCSV "#Unique words: $unique_words\n";
    print EXPCSV "#Categories:\n";
    foreach my $entry (keys %categories) {
        print EXPCSV " #- $entry: " . $categories{$entry} . " words, ". $category_uniques{$entry} . " unique words\n";
    }

    print EXPCSV "#WORD\tCATEGORY\tCOUNT\n";
    foreach my $entry (keys %words) {
        $entry =~ /^(.+)=(.+)$/; #category=word
        $category = $1;
        $word = $2;
        $count = $words{$entry};
        print EXPCSV "$word\t$category\t$count\n";
    }

    close(EXPCSV);
}

# Prune and swap the model and emit stats
sub prune_and_swap
{
    my ($thresh) = @_;
    print "OLD MODEL:\n";
    model_stats();

    prune_model( $thresh );

    untie %words;
    unlink('words_old.db');
    rename('words.db','words_old.db');
    rename('words_pruned.db','words.db');
    tie %words, 'DB_File', 'words.db';


    print "NEW MODEL:\n";
    model_stats();
}

# Supported commands are 'add' to add words to a category and
# 'classify' to get the classification of a file

if ( ( $ARGV[0] eq 'add' ) && ( $#ARGV == 2 ) ) {
    add_words( $ARGV[1], parse_file( $ARGV[2] ) );
} elsif ( ( $ARGV[0] eq 'classify' ) && ( $#ARGV == 1 ) ) {
    classify( parse_file( $ARGV[1] ) );
} elsif ( ( $ARGV[0] eq 'prune' ) && ( $#ARGV == 1 ) ) {
    prune_and_swap( $ARGV[1] );
} elsif ( ( $ARGV[0] eq 'export' ) && ( $#ARGV == 1 ) ) {
    model_export( $ARGV[1] );
} elsif ( ( $ARGV[0] eq 'stats' ) ) {
    model_stats( );
} else {
    print <<EOUSAGE;
Usage: add <category> <file> - Adds words from <file> to category <category>
       classify <file>       - Outputs classification of <file>
       prune <percentage>    - Prunes elements of model lower than frequency <X>
       stats                 - Prints stats of the model
       export <file>         - Exports model to <file>
EOUSAGE
}

untie %words;



