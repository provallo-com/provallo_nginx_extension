 
ngx_addon_name="ngx_provallo_module"
NGNIX_PROVALLO_CORE_MODULES="                                         \
                ngx_provallo_module                             \
                ngx_provallo_core_module                        \
                ngx_provallo_minidal_module                     \
                ngx_provallo_decision_engine                    \
                ngx_provallo_glue                      \
                ngx_provallo_glue_dynasm                      \
                ngx_provallo_helpers                        \
                ngx_provallo_metamon                        \
                ngx_provallo_packet_handlers                         \
                ngx_provallo_parsers                         \
                ngx_provallo_proto                     \
                ngx_provallo_raptor                      \
                ngx_provallo_rpc                      \
                ngx_provallo_statistics                   \
                ngx_provallo_third_party                      \
                ngx_provallo_util                         \
                ngx_provallo_v1_handlers                       \
                ngx_provallo_hls_module                         \
                ngx_provallo_dash_module                        \
                "


			
NGNIX_PROVALLO_HTTP_MODULES="                                         \
                ngx_provallo_stat_module                        \
                ngx_provallo_control_module                     \
                "
RTMP_DEPS="                                                 \
                $ngx_addon_dir/data/minidal.h               \
                $ngx_addon_dir/ngx_provallo_bandwidth.h         \
                $ngx_addon_dir/ngx_provallo_cmd_module.h        \
                $ngx_addon_dir/ngx_provallo_codec_module.h      \
                $ngx_addon_dir/ngx_provallo_eval.h              \
                $ngx_addon_dir/ngx_provallo.h                   \
                $ngx_addon_dir/ngx_provallo_version.h           \
                $ngx_addon_dir/ngx_provallo_live_module.h       \
                $ngx_addon_dir/ngx_provallo_netcall_module.h    \
                $ngx_addon_dir/ngx_provallo_play_module.h       \
                $ngx_addon_dir/ngx_provallo_record_module.h     \
                $ngx_addon_dir/ngx_provallo_relay_module.h      \
                $ngx_addon_dir/ngx_provallo_streams.h           \
                $ngx_addon_dir/ngx_provallo_bitop.h             \
                $ngx_addon_dir/ngx_provallo_proxy_protocol.h    \
                $ngx_addon_dir/hls/ngx_provallo_mpegts.h        \
                $ngx_addon_dir/dash/ngx_provallo_mp4.h          \
                "
NGNIX_PROVALLO_CORE_SRCS="                                            \
                $ngx_addon_dir/ngx_provallo.c                   \
                $ngx_addon_dir/ngx_provallo_init.c              \
                $ngx_addon_dir/ngx_provallo_handshake.c         \
                $ngx_addon_dir/ngx_provallo_handler.c           \
                $ngx_addon_dir/ngx_provallo_amf.c               \
                $ngx_addon_dir/ngx_provallo_send.c              \
                $ngx_addon_dir/ngx_provallo_shared.c            \
                $ngx_addon_dir/ngx_provallo_eval.c              \
                $ngx_addon_dir/ngx_provallo_receive.c           \
                $ngx_addon_dir/ngx_provallo_core_module.c       \
                $ngx_addon_dir/ngx_provallo_cmd_module.c        \
                $ngx_addon_dir/ngx_provallo_codec_module.c      \
                $ngx_addon_dir/ngx_provallo_access_module.c     \
                $ngx_addon_dir/ngx_provallo_record_module.c     \
                $ngx_addon_dir/ngx_provallo_live_module.c       \
                $ngx_addon_dir/ngx_provallo_play_module.c       \
                $ngx_addon_dir/ngx_provallo_flv_module.c        \
                $ngx_addon_dir/ngx_provallo_mp4_module.c        \
                $ngx_addon_dir/ngx_provallo_netcall_module.c    \
                $ngx_addon_dir/ngx_provallo_relay_module.c      \
                $ngx_addon_dir/ngx_provallo_bandwidth.c         \
                $ngx_addon_dir/ngx_provallo_exec_module.c       \
                $ngx_addon_dir/ngx_provallo_auto_push_module.c  \
                $ngx_addon_dir/ngx_provallo_notify_module.c     \
                $ngx_addon_dir/ngx_provallo_log_module.c        \
                $ngx_addon_dir/ngx_provallo_limit_module.c      \
                $ngx_addon_dir/ngx_provallo_bitop.c             \
                $ngx_addon_dir/ngx_provallo_proxy_protocol.c    \
                $ngx_addon_dir/hls/ngx_provallo_hls_module.c    \
                $ngx_addon_dir/dash/ngx_provallo_dash_module.c  \
                $ngx_addon_dir/hls/ngx_provallo_mpegts.c        \
                $ngx_addon_dir/dash/ngx_provallo_mp4.c          \
                "
NGNIX_PROVALLO_HTTP_SRCS="                                            \
                $ngx_addon_dir/ngx_provallo_stat_module.c       \
                $ngx_addon_dir/ngx_provallo_control_module.c    \
                "
ngx_module_incs=$ngx_addon_dir
ngx_module_deps=$RTMP_DEPS

if [ $ngx_module_link = DYNAMIC ] ; then
    ngx_module_name="$NGNIX_PROVALLO_CORE_MODULES $NGNIX_PROVALLO_HTTP_MODULES"
    ngx_module_srcs="$NGNIX_PROVALLO_CORE_SRCS $NGNIX_PROVALLO_HTTP_SRCS"
    . auto/module
elif [ $ngx_module_link = ADDON ] ; then
    ngx_module_type=CORE
    ngx_module_name=$NGNIX_PROVALLO_CORE_MODULES
    ngx_module_srcs=$NGNIX_PROVALLO_CORE_SRCS
    . auto/module
    ngx_module_type=HTTP
    ngx_module_name=$NGNIX_PROVALLO_HTTP_MODULES
    ngx_module_srcs=$NGNIX_PROVALLO_HTTP_SRCS
    . auto/module
fi

USE_OPENSSL=YES