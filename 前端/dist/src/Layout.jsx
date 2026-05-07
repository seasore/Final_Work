import { useState } from 'react'
import { useLocation, useNavigate, Outlet, NavLink } from 'react-router-dom'
import { Layout as AntLayout, Menu, Dropdown, Avatar, Space, Tabs, Button } from 'antd'
import {
  LineChartOutlined, BarChartOutlined, CalendarOutlined, AuditOutlined, DatabaseOutlined,
  SettingOutlined, UserOutlined, LogoutOutlined, KeyOutlined, DashboardOutlined,
  HeatMapOutlined, FundOutlined, FileTextOutlined, TeamOutlined,
  MenuFoldOutlined, MenuUnfoldOutlined, MessageOutlined,
} from '@ant-design/icons'

const { Header, Sider, Content } = AntLayout

const BASE_MENU_ITEMS = [
  { key: '/dashboard', icon: <DashboardOutlined />, label: '负荷预测' },
  { key: '/analysis', icon: <BarChartOutlined />, label: '负荷分析', children: [
    { key: '/analysis/correlation', icon: <HeatMapOutlined />, label: '相关性分析' },
  ]},
  { key: '/predict', icon: <LineChartOutlined />, label: '正常日预测', children: [
    { key: '/predict/manual', icon: <FundOutlined />, label: '手动预测' },
    { key: '/predict/report', icon: <FileTextOutlined />, label: '修正上报' },
  ]},
  { key: '/holiday', icon: <CalendarOutlined />, label: '节假日预测', children: [
    { key: '/holiday/analysis', icon: <BarChartOutlined />, label: '节假日特性分析' },
    { key: '/holiday/config', icon: <SettingOutlined />, label: '节假日信息管理' },
  ]},
  { key: '/evaluate', icon: <AuditOutlined />, label: '预测后评估', children: [
    { key: '/evaluate/96', icon: <HeatMapOutlined />, label: '96点算法准确率' },
    { key: '/evaluate/holiday', icon: <CalendarOutlined />, label: '节假日准确率' },
    { key: '/evaluate/report', icon: <FileTextOutlined />, label: '上报准确率' },
  ]},
  { key: '/assessment', icon: <AuditOutlined />, label: '考核管理' },
  { key: '/data', icon: <DatabaseOutlined />, label: '数据管理' },
  { key: '/reports', icon: <FileTextOutlined />, label: '我的汇报' },
  { key: '/messages', icon: <MessageOutlined />, label: '消息' },
]
const ADMIN_MENU_BASE = [
  { key: '/settings/users', icon: <UserOutlined />, label: '用户管理' },
  { key: '/settings/units', icon: <DatabaseOutlined />, label: '单位管理' },
  { key: '/settings/messages', icon: <MessageOutlined />, label: '发送消息' },
  { key: '/settings/holiday', icon: <CalendarOutlined />, label: '节假日配置' },
  { key: '/settings/model', icon: <LineChartOutlined />, label: '算法参数' },
]

export default function Layout() {
  const navigate = useNavigate()
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(false)
  const user = JSON.parse(localStorage.getItem('user') || '{}')

  const handleLogout = () => {
    localStorage.removeItem('token')
    localStorage.removeItem('user')
    navigate('/login')
  }

  const isAdmin = ['total_admin', 'group_admin'].includes(user.role)
  const isTotalAdmin = user.role === 'total_admin'
  const baseMenus = BASE_MENU_ITEMS.filter((m) => {
    if (m.key === '/reports' && isTotalAdmin) return false
    return true
  })
  const adminMenu = { key: '/settings', icon: <SettingOutlined />, label: '系统设置', children: ADMIN_MENU_BASE }
  const sidebarMenus = isAdmin ? [...baseMenus, adminMenu] : baseMenus

  const userMenu = {
    items: [
      { key: 'profile', icon: <UserOutlined />, label: user.username },
      { key: 'password', icon: <KeyOutlined />, label: '修改密码' },
      ...(isAdmin ? [{ key: 'settings', icon: <SettingOutlined />, label: '系统设置', onClick: () => navigate('/settings/users') }] : []),
      { key: 'logout', icon: <LogoutOutlined />, label: '退出登录', onClick: handleLogout },
    ],
  }

  const getSelectedKey = () => {
    const p = location.pathname
    if (p.startsWith('/analysis')) return '/analysis/correlation'
    if (p.startsWith('/predict')) return p.includes('report') ? '/predict/report' : '/predict/manual'
    if (p.startsWith('/holiday')) return p.includes('config') ? '/holiday/config' : '/holiday/analysis'
    if (p.startsWith('/evaluate')) return p.includes('holiday') ? '/evaluate/holiday' : p.includes('report') ? '/evaluate/report' : '/evaluate/96'
    if (p.startsWith('/settings')) return location.pathname || '/settings/users'
    return location.pathname || '/dashboard'
  }

  return (
    <AntLayout style={{ minHeight: '100vh' }} className="main-bg">
      <Header
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          zIndex: 1000,
          display: 'flex',
          alignItems: 'center',
          padding: '0 24px',
          height: 64,
          background: 'rgba(10, 22, 40, 0.85)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(255,255,255,0.1)',
          boxShadow: '0 4px 24px rgba(0,0,0,0.2)',
        }}
      >
        <div style={{ color: '#e8f4ff', fontSize: 18, fontWeight: 600, marginRight: 32 }}>
          短期负荷预测管理系统
        </div>
        <Dropdown menu={userMenu} placement="bottomRight" trigger={['click']}>
          <Space style={{ color: '#e8f4ff', cursor: 'pointer', marginLeft: 'auto', padding: '8px 16px', borderRadius: 8, background: 'rgba(255,255,255,0.05)' }}>
            <Avatar size="small" icon={<UserOutlined />} style={{ background: 'rgba(24,144,255,0.5)' }} />
            <span>{user.username}</span>
            <span style={{ color: 'rgba(255,255,255,0.6)', fontSize: 12 }}>
              ({user.role === 'total_admin' ? '超级管理员' : user.role === 'group_admin' ? '系统管理员' : '预测专员'})
            </span>
          </Space>
        </Dropdown>
      </Header>

      <AntLayout style={{ marginTop: 64 }}>
        <Sider
          collapsed={collapsed}
          width={220}
          collapsedWidth={80}
          style={{
            position: 'fixed',
            left: 0,
            top: 64,
            bottom: 0,
            zIndex: 999,
            overflow: 'auto',
            background: 'linear-gradient(180deg, rgba(10, 22, 40, 0.95) 0%, rgba(26, 58, 92, 0.9) 100%)',
            borderRight: '1px solid rgba(255,255,255,0.08)',
          }}
        >
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{ color: '#fff', margin: '16px auto', display: 'block' }}
          />
          <Menu
            mode="inline"
            selectedKeys={[getSelectedKey()]}
            defaultOpenKeys={['/analysis', '/predict', '/holiday', '/evaluate', ...(isAdmin ? ['/settings'] : [])]}
            style={{ height: 'calc(100% - 60px)', borderRight: 'none', background: 'transparent', color: '#e8f4ff' }}
            items={sidebarMenus.map((m) =>
              m.children
                ? {
                    key: m.key,
                    icon: m.icon,
                    label: m.label,
                    children: m.children.map((c) => ({
                      key: c.key,
                      icon: c.icon,
                      label: <NavLink to={c.key}>{c.label}</NavLink>,
                    })),
                  }
                : { key: m.key, icon: m.icon, label: <NavLink to={m.key}>{m.label}</NavLink> }
            )}
          />
        </Sider>

        <Content style={{ marginLeft: collapsed ? 80 : 220, padding: 24, minHeight: 'calc(100vh - 64px)' }}>
          <div className="glass-card" style={{ padding: 24, minHeight: '100%' }}>
            <Outlet />
          </div>
        </Content>
      </AntLayout>
    </AntLayout>
  )
}
