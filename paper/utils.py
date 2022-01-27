import os
import paramiko
from scp import SCPClient

def get_healpixel_files(cfg, nside, pixels):
    def createSSHClient(server, port, user, password):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, port, user, password)
        return client

    datadir = cfg['catalog']['dirname']
    
    exists = []
    for pixel in pixels:
        if os.path.exists('{}/y6_gold_2_0_{:0>5n}.fits'.format(datadir, pixel)):
            exists.append(pixel)

    if len(exists) < len(pixels):
        print("Copying over {} files".format(len(pixels) - len(exists)))
        ssh = createSSHClient('login.hep.wisc.edu', 22, 'mcnanna', 'P!llar20')
        scp = SCPClient(ssh.get_transport())
        
        for i, pixel in enumerate(pixels):
            if pixel not in exists:
                print "File {} ...".format(i+1)
                try:
                    scp.get('~/data/skim_y6_gold_2_0/y6_gold_2_0_{:0>5n}.fits'.format(pixel), '{}/y6_gold_2_0_{:0>5n}.fits'.format(datadir, pixel))
                except Exception as e:
                    print(e)
        scp.close()
        ssh.close()
