/*
 * netlinkstatshelper.h
 *
 *  Created on: Jun 26, 2021
 *      Author: kardon
 */

#ifndef HELPERS_NETLINKSTATSHELPER_H_
#define HELPERS_NETLINKSTATSHELPER_H_
#include <errno.h>
#include <error.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <linux/if.h>
#include <linux/netlink.h>
#include <linux/netdevice.h>
#include <linux/rtnetlink.h>
#include <iostream>
#include <csignal>
#include <thread>
#include <condition_variable>
#include <map>
#include <vector>
#include "../data/minidal.h"
#include "../decision_engine/classifier.h"

#include "../statistics/statcollector.h"
#include <chrono>

extern provallo::classifier *_lastclassifier;

namespace provallo
{
#define NL_LINK_BUF_SIZE 16384

	extern isolation_forest *iso_last_fit;

	class netlink_stats_helper
	{

		bool _valid;
		int _socket;
		bool _running;
		struct nlsock_link_stats
		{
			int fd;
			int dev_id;
			struct
			{
				struct nlmsghdr n;
				struct ifinfomsg r;
			} req;
			char buf[NL_LINK_BUF_SIZE];
			size_t len;
		} *_pstats;

		proc_collector *_collector;
		provallo::mini_dal *_pDal;

		std::vector<provallo::stat_collector *> _collectors;
		std::chrono::steady_clock _wait_for;

	public:
		explicit netlink_stats_helper(provallo::mini_dal *pDal,
									  const std::vector<provallo::stat_collector *> &rcoll, proc_collector *_c = nullptr) : _valid(false), _socket(0), _running(true), _pstats(nullptr), _collector(_c), _pDal(pDal), _collectors(rcoll)
		{
		}

	protected:
		explicit netlink_stats_helper() : _valid(false), _running(true), _pstats(nullptr)
		{
			struct sockaddr_nl addr;
			struct nlmsghdr *nlmp;
			struct ifinfomsg *rtmp;
			struct rtattr *rtatp;

			_pstats = new nlsock_link_stats;

			//_socket = socket(AF_NETLINK, SOCK_RAW, NETLINK_ROUTE);
			_socket = socket(PF_NETLINK, SOCK_DGRAM | SOCK_NONBLOCK, NETLINK_ROUTE);
			_pstats->fd = _socket;
			struct nlsock_link_stats *nlsk = this->_pstats;
			if (_socket > 0)
			{
				::memset(&addr, 0, sizeof(addr));
				addr.nl_family = AF_NETLINK;
				addr.nl_pid = getpid();
				addr.nl_groups = RTM_GETLINK;
				if (bind(_socket, (struct sockaddr *)&addr, sizeof(addr)) > -1)
				{
					_valid = true;
				}
			}
			if (!_valid)
			{
				std::cerr << "[-] " << strerror(errno) << " Check permissions... ";
			}

			int res = link_read();

			size_t pos = nlsk->len;
			// Loop MESSAGES //

			if (res > 0)
			{
				for (nlmp = (struct nlmsghdr *)nlsk->buf; pos > sizeof(*nlmp);)
				{
					size_t msg_len = nlmp->nlmsg_len;
					size_t req_len = msg_len - sizeof(*nlmp);

					if (int(req_len) < 0 || msg_len > pos)
					{
						break;
					}
					if (!NLMSG_OK(nlmp, pos))
					{
						break;
					}

					rtmp = (struct ifinfomsg *)NLMSG_DATA(nlmp);
					rtatp = (struct rtattr *)IFLA_RTA(rtmp);
					int rtattrlen = IFLA_PAYLOAD(nlmp);

					if (rtmp->ifi_index == 0)
					{
						break;
					}
					// printf("Index Of Iface= %d, %d\n",rtmp->ifi_index,rtattrlen);
					int dev_id = rtmp->ifi_index;

					// LOOP ATTRIBUTES //
					for (; RTA_OK(rtatp, rtattrlen); rtatp = RTA_NEXT(rtatp, rtattrlen))
					{
						if (rtatp->rta_type == IFLA_IFNAME)
						{
							_pstats->dev_id = dev_id; // keep the last dev id
						}
					}
					pos -= NLMSG_ALIGN(msg_len);
					nlmp = (struct nlmsghdr *)((char *)nlmp + NLMSG_ALIGN(msg_len));
				}
			}
			this->_running = true;
		}

	public:
		int
		link_read()
		{

			struct nlsock_link_stats *nlsk = this->_pstats;

			if (nlsk == nullptr)
				return 0;
			memset(&nlsk->req, 0, sizeof(nlsk->req));
			nlsk->req.n.nlmsg_len = NLMSG_LENGTH(sizeof(struct rtgenmsg));
			nlsk->req.n.nlmsg_flags = NLM_F_REQUEST | NLM_F_ROOT;
			nlsk->req.n.nlmsg_type = RTM_GETLINK;
			nlsk->req.n.nlmsg_pid = getpid();

			int status;

			// SEND DUMP ROOT REQUEST //
			status = send(nlsk->fd, &nlsk->req, nlsk->req.n.nlmsg_len, 0);
			if (status < 0)
			{
				return 0;
			}

			// RECV DUMP ROOT //
			int len = status = 0;
			do
			{
				len += status;
				status = recv(nlsk->fd, &nlsk->buf[len], sizeof(nlsk->buf) - len, 0);
				if (status > 0 && len + status > NL_LINK_BUF_SIZE)
				{

					std::cerr << "[-] netlink stats buffer explosion";

					return 0;
				}
			} while (status > 0);

			nlsk->len = len;
			return 1;
		}
		std::map<uint16_t, struct rtnl_link_stats>
		get_stats()
		{
			// int status = 0;
			struct nlmsghdr *nlmp;
			struct ifinfomsg *rtmp;
			struct rtattr *rtatp;
			struct nlsock_link_stats *nlsk = this->_pstats;
			struct rtnl_link_stats stats;
			std::map<uint16_t, struct rtnl_link_stats> dev_stats;
			int pos = nlsk->len;
			for (nlmp = (struct nlmsghdr *)nlsk->buf; pos > 0 && size_t(pos) > sizeof(*nlmp);)
			{
				int len = nlmp->nlmsg_len;
				int req_len = len - sizeof(*nlmp);
				if (req_len < 0 || len > pos)
				{
					break;
				}
				if (!NLMSG_OK(nlmp, pos))
				{
					break;
				}

				rtmp = (struct ifinfomsg *)NLMSG_DATA(nlmp);
				rtatp = (struct rtattr *)IFLA_RTA(rtmp);
				int rtattrlen = IFLA_PAYLOAD(nlmp);

				// printf("Index stats= %d, %d\n",rtmp->ifi_index,rtattrlen);
				if (rtmp->ifi_index == 0)
				{
					break;
				}
				// LOOP ATTRIBUTES //
				for (; RTA_OK(rtatp, rtattrlen); rtatp = RTA_NEXT(rtatp, rtattrlen))
				{
					if (rtatp->rta_type == IFLA_STATS)
					{
						struct rtnl_link_stats *link_stats =
							(struct rtnl_link_stats *)RTA_DATA(rtatp);
						memcpy(&stats, link_stats, sizeof(struct rtnl_link_stats));
						dev_stats.insert(
							std::pair<int, struct rtnl_link_stats>(rtmp->ifi_index,
																   stats));
					}
				}
				pos -= NLMSG_ALIGN(len);
				nlmp = (struct nlmsghdr *)((char *)nlmp + NLMSG_ALIGN(len));
			}

			return dev_stats;
		}
		std::map<uint16_t, struct rtnl_link_stats64>
		get_stats64()
		{
			struct nlmsghdr *nlmp;
			struct ifinfomsg *rtmp;
			struct rtattr *rtatp;
			struct nlsock_link_stats *nlsk = this->_pstats;
			struct rtnl_link_stats64 stats;
			std::map<uint16_t, struct rtnl_link_stats64> dev_stats;

			if (!nlsk)
				return dev_stats;
			size_t pos = nlsk->len;
			for (nlmp = (struct nlmsghdr *)nlsk->buf; pos > sizeof(*nlmp);)
			{
				size_t len = nlmp->nlmsg_len;
				size_t req_len = len - sizeof(*nlmp);
				if (int(req_len) < 0 || len > pos)
				{
					break;
				}
				if (!NLMSG_OK(nlmp, pos))
				{
					break;
				}

				rtmp = (struct ifinfomsg *)NLMSG_DATA(nlmp);
				rtatp = (struct rtattr *)IFLA_RTA(rtmp);
				int rtattrlen = IFLA_PAYLOAD(nlmp);

				// printf("Index stats= %d, %d\n",rtmp->ifi_index,rtattrlen);
				if (rtmp->ifi_index == 0)
				{
					break;
				}
				// LOOP ATTRIBUTES //
				for (; RTA_OK(rtatp, rtattrlen); rtatp = RTA_NEXT(rtatp, rtattrlen))
				{
					if (rtatp->rta_type == IFLA_STATS64)
					{
						struct rtnl_link_stats64 *link_stats =
							(struct rtnl_link_stats64 *)RTA_DATA(rtatp);
						memcpy(&stats, link_stats, sizeof(struct rtnl_link_stats64));
						dev_stats.insert(
							std::pair<int, struct rtnl_link_stats64>(rtmp->ifi_index,
																	 stats));
					}
				}
				pos -= NLMSG_ALIGN(len);
				nlmp = (struct nlmsghdr *)((char *)nlmp + NLMSG_ALIGN(len));
			}

			return dev_stats;
		}
		void
		classify_stats()
		{


			static std::string lastdecision = "normal";


			std::cout << "[+] classify_stats thread started";

			netlink_stats_helper *_thisctx = this;
			while (_thisctx->_running)
			{
				using namespace std::literals::chrono_literals;
				auto start = std::chrono::high_resolution_clock::now();
				std::cout << "[+] testing classifier";
				try
				{

					if (_lastclassifier)
					{

						std::cout << *_lastclassifier;
						std::string normalized;
						_thisctx->_collector->collect();

						// trying to classify last sample  :
						if (false)
						{
							normalized = _thisctx->_collector->get_last_sample();

							if (normalized.length() > 10)
							{
								std::cout << std::string("[+] last sample string : ") + normalized << std::endl;

								class_dist distribution = _lastclassifier->posterior(normalized);
								attribute result = distribution.mode();
								float prob = distribution.percentage(result.discrete());
								std::string decision = _lastclassifier->getClassName(result);
								std::cout << "[+]  classifier identified system state as: [" << decision << "] with probability = " << prob << std::endl;
								std::cout << distribution << std::endl;
							}
						}
						else
						{
							normalized = _thisctx->_collector->get_normalized();

							if (normalized.length() > 10)
							{
								std::cout << std::string("[+] normalized string : ") + normalized << std::endl;

								class_dist distribution = _lastclassifier->posterior(normalized);
								attribute result = distribution.mode();
								float prob = distribution.percentage(result.discrete());
								std::string decision = _lastclassifier->getClassName(result);
								std::cout << "[+]  classifier identified system state as: [" << decision << "] with probability = " << prob << std::endl;
								std::cout << distribution << std::endl;
							}
						}
						if (iso_last_fit != nullptr)
						{
							std::vector<double> result = iso_last_fit->predict(normalized);
							std::cout << "[+] iso forest returned :" << std::endl;
							for (auto res : result)
								std::cout << "[+] [+] " << std::to_string(res) << std::endl;
						}
					}
					else
					{
						std::cerr << "[=] last classifier is not initialized." << std::endl;
					}
				}
				catch (std::exception &ex)
				{
					std::cerr << "[=] exception classifying state : " << ex.what() << std::endl;
				}

				auto stop = std::chrono::high_resolution_clock::now();
				auto sleeptime = 4000ms;
				sleeptime = sleeptime - std::chrono::duration_cast<std::chrono::milliseconds>(
											start - stop);
				std::this_thread::sleep_for(sleeptime);
			}
		}

		void
		collect_stats()
		{
			netlink_stats_helper *_thisctx = this;
			while (_thisctx->_running)
			{
				_thisctx->link_read();
				// ignore 32bit stats

#if 0
			auto stats = _thisctx->get_stats();
			for (auto it : stats)
			{
				std::cout<<"stats for interface :"<<it.first;
				std::cout<<" collisions:"<<it.second.collisions;
				std::cout<<",rx_bytes:"<<it.second.rx_bytes;
				std::cout<<",rx_compressed:"<<it.second.rx_compressed;
				std::cout<<",rx_crc_errors:"<<it.second.rx_crc_errors;
				std::cout<<",rx_dropped:"<<it.second.rx_dropped;
				std::cout<<",rx_errors:"<<it.second.rx_errors;
				std::cout<<",rx_fifo_errors:"<<it.second.rx_fifo_errors;
 				std::cout<<",rx_frame_errors:"<<it.second.rx_frame_errors;
				std::cout<<",rx_length_errors:"<<it.second.rx_length_errors;
				std::cout<<",rx_missed_errors"<<it.second.rx_missed_errors;
				std::cout<<",rx_nohandler:"<<it.second.rx_nohandler;
				std::cout<<",rx_over_errors:"<<it.second.rx_over_errors;
				std::cout<<",rx_packets:"<<it.second.rx_packets;
				std::cout<<",tx_aborted_errors:"<<it.second.tx_aborted_errors;
			    std::cout<<",tx_bytes:"<<it.second.tx_bytes;
			    std::cout<<",tx_carrier_errors:"<<it.second.tx_carrier_errors;
			    std::cout<<",tx_compressed:"<<it.second.tx_compressed;
			    std::cout<<",tx_dropped:"<<it.second.tx_dropped;
			    std::cout<<",tx_errors:"<<it.second.tx_errors;
			    std::cout<<",tx_fifo_errors:"<<it.second.tx_fifo_errors;
			    std::cout<<",tx_heartbeat_errors:"<<it.second.tx_heartbeat_errors;
			    std::cout<<",tx_packets:"<<it.second.tx_packets;
			    std::cout<<",tx_window_errors:"<<it.second.tx_window_errors;
			    std::cout<<" ";

			}


			//comment printouts
			for (auto it : stats64)
						{
							std::cout<<"stats64 for interface :"<<it.first;
							std::cout<<" collisions64:"<<it.second.collisions;
							std::cout<<",rx_bytes64:"<<it.second.rx_bytes;
							std::cout<<",rx_compressed64:"<<it.second.rx_compressed;
							std::cout<<",rx_crc_errors64:"<<it.second.rx_crc_errors;
							std::cout<<",rx_dropped64:"<<it.second.rx_dropped;
							std::cout<<",rx_errors64:"<<it.second.rx_errors;
							std::cout<<",rx_fifo_errors64:"<<it.second.rx_fifo_errors;
			 				std::cout<<",rx_frame_errors64:"<<it.second.rx_frame_errors;
							std::cout<<",rx_length_errors64:"<<it.second.rx_length_errors;
							std::cout<<",rx_missed_errors64:"<<it.second.rx_missed_errors;
							std::cout<<",rx_nohandler64:"<<it.second.rx_nohandler;
							std::cout<<",rx_over_errors64:"<<it.second.rx_over_errors;
							std::cout<<",rx_packets64:"<<it.second.rx_packets;
							std::cout<<",tx_aborted_errors64:"<<it.second.tx_aborted_errors;
						    std::cout<<",tx_bytes64:"<<it.second.tx_bytes;
						    std::cout<<",tx_carrier_errors64:"<<it.second.tx_carrier_errors;
						    std::cout<<",tx_compressed64:"<<it.second.tx_compressed;
						    std::cout<<",tx_dropped64:"<<it.second.tx_dropped;
						    std::cout<<",tx_errors64:"<<it.second.tx_errors;
						    std::cout<<",tx_fifo_errors64:"<<it.second.tx_fifo_errors;
						    std::cout<<",tx_heartbeat_errors64:"<<it.second.tx_heartbeat_errors;
						    std::cout<<",tx_packets64:"<<it.second.tx_packets;
						    std::cout<<",tx_window_errors64:"<<it.second.tx_window_errors;
						    std::cout<<" ";

						}

#endif
				using namespace std::literals::chrono_literals;
				auto start = std::chrono::high_resolution_clock::now();
				auto stats64 = _thisctx->get_stats64();

				if (this->_pDal)
				{

					_pDal->report_link_statistics(stats64);
					/*for (auto coll : _collectors)
				  {
					coll->collect ();
					//don't insert anything to db periodically.

					//std::string ins = coll->insert_query ();
					//_pDal->run_query (ins);
				  }
				  */

					// classify attributes :
					//
					auto stop = std::chrono::high_resolution_clock::now();
					auto sleeptime = 4000ms;
					sleeptime = sleeptime - std::chrono::duration_cast<std::chrono::milliseconds>(
												start - stop);
					std::this_thread::sleep_for(sleeptime);
				}
			}
		}

		void
		link_release()
		{
			if (_pstats)
			{
				close(_pstats->fd);
				delete _pstats;
				_pstats = nullptr;
			}
		}
		virtual ~netlink_stats_helper()
		{
			link_release();
		}
	};

} /* namespace provallo */

#endif /* HELPERS_NETLINKSTATSHELPER_H_ */
