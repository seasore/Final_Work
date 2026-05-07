import { useState, useEffect } from 'react'
import { Table, Button, Modal, Form, Input, Select, Switch, message } from 'antd'
import { adminApi } from '../../api'

export default function AdminUsers() {
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [modalOpen, setModalOpen] = useState(false)
  const [form] = Form.useForm()

  const load = async () => {
    setLoading(true)
    try {
      const res = await adminApi.listUsers()
      setUsers(res.data || [])
    } catch (e) {
      message.error('加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const handleDisable = async (userId, disabled) => {
    try {
      await adminApi.disableUser(userId, disabled)
      message.success(disabled ? '已禁用' : '已启用')
      load()
    } catch (e) {
      message.error(e.response?.data?.detail || '操作失败')
    }
  }

  const handleCreate = async () => {
    try {
      const v = await form.validateFields()
      await adminApi.createUser(v)
      message.success('创建成功')
      setModalOpen(false)
      form.resetFields()
      load()
    } catch (e) {
      if (e.errorFields) return
      message.error(e.response?.data?.detail || '创建失败')
    }
  }

  const currentUser = JSON.parse(localStorage.getItem('user') || '{}')
  const isTotalAdmin = currentUser.role === 'total_admin'

  const handleSetRole = async (userId, role) => {
    try {
      await adminApi.updateUserRole(userId, role)
      message.success(role === 'group_admin' ? '已任命为小组管理员' : '已撤销小组管理员')
      load()
    } catch (e) {
      message.error(e.response?.data?.detail || '操作失败')
    }
  }

  const columns = [
    { title: '用户名', dataIndex: 'username' },
    { title: '角色', dataIndex: 'role', render: (r) => ({ total_admin: '总管理员', group_admin: '小组管理员', member: '成员' }[r] || r) },
    { title: '小组', dataIndex: 'group_id' },
    {
      title: '状态',
      dataIndex: 'disabled',
      render: (d, r) => {
        const canDisable = isTotalAdmin || (r.group_id === currentUser.group_id && r.role !== 'total_admin')
        return canDisable ? <Switch checked={!d} onChange={(v) => handleDisable(r.id, !v)} /> : <span>{d ? '已禁用' : '正常'}</span>
      },
    },
    ...(isTotalAdmin ? [{
      title: '任命/撤销',
      render: (_, r) => {
        if (r.role === 'total_admin') return '-'
        return r.role === 'group_admin'
          ? <Button type="link" size="small" onClick={() => handleSetRole(r.id, 'member')}>撤销小组管理员</Button>
          : <Button type="link" size="small" onClick={() => handleSetRole(r.id, 'group_admin')}>任命小组管理员</Button>
      },
    }] : []),
  ]

  return (
    <div>
      <div style={{ marginBottom: 16 }}>
        {isTotalAdmin && <Button type="primary" onClick={() => setModalOpen(true)}>创建用户</Button>}
      </div>
      <Table columns={columns} dataSource={users} rowKey="id" loading={loading} />
      <Modal title="创建用户" open={modalOpen} onOk={handleCreate} onCancel={() => setModalOpen(false)}>
        <Form form={form} layout="vertical">
          <Form.Item name="username" label="用户名" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item name="password" label="密码" rules={[{ required: true }]}>
            <Input.Password />
          </Form.Item>
          <Form.Item name="role" label="角色" rules={[{ required: true }]}>
            <Select options={[
              { value: 'member', label: '成员' },
              { value: 'group_admin', label: '小组管理员' },
              { value: 'total_admin', label: '总管理员' },
            ]} />
          </Form.Item>
          <Form.Item name="group_id" label="小组ID">
            <Input placeholder="小组管理员和成员需填写" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}
