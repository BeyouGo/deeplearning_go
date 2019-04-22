# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import multiprocessing
import six
if sys.version_info[0] == 3:
    from urllib.request import urlopen, urlretrieve
else:
    from urllib import urlopen, urlretrieve


def worker(url_and_target):
    '''
    Parallelize data download via multiprocessing
    '''
    try:
        (url, target_path) = url_and_target
        print('>>> Downloading ' + target_path)
        urlretrieve(url, target_path)
    except (KeyboardInterrupt, SystemExit):
        print('>>> Exiting child process')


class OGSIndex(object):

    def __init__(self,
                 kgs_url='localhost',
                 index_page='ogs_index-9x9.html',
                 data_directory='data_9x9'):
        '''
        Create an index of zip files containing SGF data of actual Go Games on KGS.

        Parameters:
        -----------
        kgs_url: URL with links to zip files of games
        index_page: Name of local html file of kgs_url
        data_directory: name of directory relative to current path to store SGF data
        '''
        self.kgs_url = kgs_url
        self.index_page = index_page
        self.data_directory = data_directory
        self.file_info = []
        self.urls = []
        # self.load_index()  # Load index on creation
        #
        # self.file_info.append(["OGS-2019-01-9x9.tar.gz",1020])
        # self.file_info.append(["OGS-2019-02-9x9.tar.gz",1070])
        # self.file_info.append(["OGS-2019-03-9x9.tar.gz",1202])
        # self.file_info.append(["OGS-2019-04-9x9.tar.gz",1075])
        # self.file_info.append(["OGS-2019-05-9x9.tar.gz",1056])

        # filenames = ['OGS-2019-01-9x9.tar.gz','OGS-2019-02-9x9.tar.gz','OGS-2019-03-9x9.tar.gz','OGS-2019-04-9x9.tar.gz','OGS-2019-05-9x9.tar.gz']
        filenames = ['OGS-2019-01-9x9-2500.tar.gz','OGS-2019-02-9x9-2500.tar.gz','OGS-2019-03-9x9-2500.tar.gz','OGS-2019-04-9x9-2500.tar.gz','OGS-2019-05-9x9-2500.tar.gz']
        # num_games =  [1020,1070,1202,1075,1056]
        # filenames = ['OGS-2019-01-9x9-2500.tar.gz']
        num_games = [444,456,422,387,446]

        # data = {}
        # data['filename'] = 'OGS-2019-01-9x9.tar.gz'
        # data['num_games'] = 1020

        for index in range(0,len(filenames)):
            data = {}
            data['filename'] = filenames[index]
            data['num_games'] = num_games[index]
            self.file_info.append(data)



        print(self.file_info)




    def download_files(self):
        '''
        Download zip files by distributing work on all available CPUs
        '''
        if not os.path.isdir(self.data_directory):
            os.makedirs(self.data_directory)

        urls_to_download = []
        for file_info in self.file_info:
            url = file_info['url']
            file_name = file_info['filename']
            if not os.path.isfile(self.data_directory + '/' + file_name):
                urls_to_download.append((url, self.data_directory + '/' + file_name))
        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        try:
            it = pool.imap(worker, urls_to_download)
            for i in it:
                pass
            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print(">>> Caught KeyboardInterrupt, terminating workers")
            pool.terminate()
            pool.join()
            sys.exit(-1)

    def create_index_page(self):
        '''
        If there is no local html containing links to files, create one.
        '''
        if os.path.isfile(self.index_page):
            print('>>> Reading cached index page')
            index_file = open(self.index_page, 'r')
            index_contents = index_file.read()
            index_file.close()
        else:
            print('>>> Downloading index page')
            fp = urlopen(self.kgs_url)
            data = six.text_type(fp.read())
            fp.close()
            index_contents = data
            index_file = open(self.index_page, 'w')
            index_file.write(index_contents)
            index_file.close()
        return index_contents

    def load_index(self):
        '''
        Create the actual index representation from the previously downloaded or cached html.
        '''
        index_contents = self.create_index_page()
        split_page = [item for item in index_contents.split('<a href="') if item.startswith("https://")]
        for item in split_page:
            download_url = item.split('">Download')[0]
            if download_url.endswith('.tar.gz'):
                self.urls.append(download_url)
        for url in self.urls:
            filename = os.path.basename(url)
            split_file_name = filename.split('-')
            num_games = int(split_file_name[len(split_file_name) - 2])
            print(filename + ' ' + str(num_games))
            self.file_info.append({'url': url, 'filename': filename, 'num_games': num_games})


if __name__ == '__main__':
    index = OGSIndex()
    index.download_files()
